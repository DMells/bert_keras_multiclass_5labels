from dotenv import find_dotenv, load_dotenv
import tensorflow as tf
import subprocess
load_dotenv(find_dotenv())


if __name__=='__main__':
    cmd = 'tensorboard --logdir s3://sn-classification/Tenders/Logs/'
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
