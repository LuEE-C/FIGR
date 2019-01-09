import os
from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import pickle as pkl
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistMetaEnv:
    def __init__(self, height=28, length=28):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.MNIST(root='./data', train=True, download=True)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.height, self.length))
        self.sample_training_task()

    def sample_training_task(self, batch_size=64):
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]), dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]), dtype=torch.float, device=device)
        return batch, task

    def make_tasks(self):
        self.task_to_examples = {}
        self.all_tasks = set(self.data.train_labels.numpy())
        for i, digit in enumerate(self.data.train_labels.numpy()):
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def split_validation_and_training_task(self):
        self.validation_task = {9}
        self.training_task = self.all_tasks - self.validation_task


class OmniglotMetaEnv:
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.Omniglot(root='./data', download=True)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.resize = transforms.Resize((self.height, self.length))
        self.to_tensor = transforms.ToTensor()

    def sample_training_task(self, batch_size=4):
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]), dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]), dtype=torch.float, device=device)
        return batch, task

    def make_tasks(self):
        self.task_to_examples = {}
        self.all_tasks = set()
        for i, (_, digit) in enumerate(self.data):
            self.all_tasks.update([digit])
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 20))
        self.training_task = self.all_tasks - self.validation_task


class FIGR8MetaEnv(Dataset):
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.resize = transforms.Resize((height, length))
        self.to_tensor = transforms.ToTensor()

        self.tasks = self.get_tasks()
        self.all_tasks = set(self.tasks)
        self.split_validation_and_training_task()

    def get_tasks(self):
        if os.path.exists('./data/FIGR-8') is False:
            if os.path.exists('./data') is False:
                os.mkdir('./data')
            os.mkdir('./data/FIGR-8')
            from google_drive_downloader import GoogleDriveDownloader as gdd
            gdd.download_file_from_google_drive(file_id='10dF30Qqi9RdIUmET9fBhyeRN0hJmq7pO',
                                                dest_path='./data/FIGR-8/Data.zip')
            import zipfile
            with zipfile.ZipFile('./data/FIGR-8/Data.zip', 'r') as zip_f:
                zip_f.extractall('./data/FIGR-8/')
            os.remove('./data/FIGR-8/Data.zip')


        tasks = dict()
        path = './data/FIGR-8/Data'
        for task in os.listdir(path):
            tasks[task] = []
            task_path = os.path.join(path, task)
            for imgs in os.listdir(task_path):
                img = Image.open(os.path.join(task_path, imgs))
                tasks[task].append(np.array(self.to_tensor(self.resize(img))))
            tasks[task] = np.array(tasks[task])
        return tasks

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 50))
        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open('validation_task.pkl', 'wb'))
        pkl.dump(self.training_task, open('training_task.pkl', 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def __len__(self):
        return len(self.files)
