
---


# ISL (Indian Sign Language) Recognition using PyTorch

This project focuses on building a deep learning model for Indian Sign Language (ISL) recognition using PyTorch. It processes large video datasets and trains a multi-class classification model using spatial-temporal features extracted from sign videos.

---

## 🛠️ Setup Instructions

### 1. Install Requirements

#### Install `wget` (if not already installed)

```bash
brew install wget
```


---

### 2. Download Dataset (56 GB)

```bash
cd Data
bash download_data.sh
```

> ⏳ **Note**: This dataset is large (\~56 GB) and may take time to download.

Once completed, return to the project root:

```bash
cd ..
```

---

### 3. Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see your terminal prefixed like:

```
(venv) your-machine-name %
```

---

### 4. Set Python Interpreter in VS Code

* Press `Command + Shift + P`
* Search for **"Python: Select Interpreter"**
* Choose the one that starts with `.venv` (the virtual environment you just created)

---

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### 6. .env File

```bash
echo 'HF_TOKEN=' > .env
```
* Go to huggingFace -> signIn and then create a new token from the following: settings -> Access Tokens -> + Create New Token
* Assign HF_TOKEN variable present int the .env file with this created token

---


### 7. Start Training

```bash
python3 pytorch_main.py
```

---

## 📂 Project Structure Overview

| File                         | Description                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| `pytorch_main.py`            | Main entry point for training the model                                               |
| `config.py`                  | Configuration (e.g., epochs, image size, paths, batch size)                           |
| `pytorchDatapreprocessor.py` | Processes videos into frames, extracts landmarks using MediaPipe, creates DataLoaders |
| `pytorchModelBuilder.py`     | Defines the model architecture                                                        |
| `pytorchTrainer.py`          | Handles training, validation, evaluation, logging, and checkpointing                  |
| `logger.py`                  | Custom logger to save logs to a timestamped file                                      |
| `download_data.sh`           | Shell script to download all dataset parts and unzip them                             |

---




# Project overflow -> how to understand the whole codebase

* In the pytorch_main.py file, ISLConfig is called first which is present in config.py. This contains all the configuration infos, eg: number of epochs, image size, number of frames, directory names etc.
* Next Datapreprocessor is called from pytorch_dataPreprocessor.py file. This is responsible for changing videos into frames, gathering landmarks using mediapipe, processing videos, creating DataLoader so that our videos are called into memory efficiently
* Note that in main file, first we called the load_and_preprocess_data() function. This function will change the 56 GB data into the desried chunks and will store the data in a new repository. After this function is complete you will be able to see a new folder and all the data will be stored as chunk. Note that at the end of this folder two new files manifest and label encoder are created, it is very important to have them.

 **`load_and_preprocess_data()`** (in `pytorchDatapreprocessor.py`)

   * Converts videos into frame sequences.
   * Extracts landmarks using MediaPipe.
   * Splits and saves data chunks.
   * Generates two important files:

     * `manifest.pkl`
     * `label_encoder.pkl`

   ✅ **First Run**:

   ```python
   manifest, class_names = preprocessor.load_and_preprocess_data()
   num_classes = len(class_names)
   config.NUM_CLASSES = num_classes
   print(f"Number of classes: {config.NUM_CLASSES}")
   ```


* now after that, we called load_data_for_training function. This function creates returns train and test dataloaders from the newly formed datachunks which are being stored. Please comment the following once the chunks are being downloaded successfully


  ```python
   # manifest, class_names = preprocessor.load_and_preprocess_data()
   # num_classes = len(class_names)
   # config.NUM_CLASSES = num_classes
   # print(f"Number of classes: {config.NUM_CLASSES}")
   ```

* As you dont want to keep downloading the same thing again and again, comment this and just call the load_data_for_training function next time
**But after commenting, you need to manually fill the following

  ```python
   # manifest, class_names = preprocessor.load_and_preprocess_data()
   # num_classes = len(class_names)
   config.NUM_CLASSES = <write in manually>
   print(f"Number of classes: {config.NUM_CLASSES}")
   ```

* You must have found this value in the print in your terminal when you first time called it during load_and_preprocess_data()


* Once we are returned the train and test loader, we called the build model function through the pytorchModelBuilder.py file. Lets not go into the architecture if this file right now. Just assume that this file has created our final model and we just need to move ahead and train this model now
* Next we called the trainer function from the pytorchTrainer.py,we brought our model on the device -> mps (fast for mac than cpu but need to use carefully as memory is limited, RAM)
* Here first of all train function is called, and inside the loop it calls train_epoch function for a total of "epoch" number of times, I have set that epoc value to 100 for now
* Any time the memory gets filled, cleanup_memory() function is called and that iteration is called to be processed again


## 🧹 Memory Management & Optimization

If you get repeated errors like:

```
OOM error at batch 4, cleaning up memory and retrying...
```

This means your model or batch size is too large for your Mac’s RAM.

### ✅ Try the following:

* Reduce `BATCH_SIZE` in `config.py`
* Reduce input dimensions (image size, number of frames)
* Restart the training after adjusting the above

> 🧠 MacBook M2 Air (8 GB/16 GB RAM) may not be sufficient for high-batch video training. Optimize the configuration accordingly.

---

## ⚠️ Troubleshooting

* **Memory errors (MPS):** Try smaller batch sizes (`BATCH_SIZE = 2` or even `1`)
* **Slow training:** MPS is fast for small models but may bottleneck due to RAM constraints.
* **Logs not updating?** Check `logs/` directory for timestamped `.log` files.

---

```
  Now this is where I am handing of the task, The batch size is initially set to 4 in config file, if you keep getting he OOM error even after cleanup this means you will need to customise the size according to your hardware, try reducting batch size and see what suits you, reducing the input length will help as well.
  My laptop is stuck here, need a better hardware now. Just follow the instructions and see what works best
```

```
  If you need some help understanding this, please let me know

```

  

