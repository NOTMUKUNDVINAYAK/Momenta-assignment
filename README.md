# Momenta Audio Deepfake Detection Take-Home Assessment

## Overview
This repo is my submission for the Momenta Audio Deepfake Detection Take-Home Assessment. I’ve built a spectrogram-based Convolutional Neural Network (CNN) to spot AI-generated human speech, using a slice of the CD-ADD dataset. My goal was to dig into some promising techniques, put together a workable solution, and share my process with a bit of analysis. I hope you find it interesting!

## Part 1: Research & Selection

### 1. AASIST (Attention-based Anti-Spoofing with Raw Waveform)
- **Innovation**: This approach works directly with raw audio waveforms and uses an attention mechanism to zero in on the trickiest parts that might reveal a deepfake—no fancy feature engineering needed!
- **Metrics**: It’s got an Equal Error Rate (EER) of about 2-5% on the ASVspoof 2019 dataset.
- **Why I Like It**: It feels promising for real-time use and could handle real conversations or AI-generated speech really well thanks to that attention focus.
- **Downsides**: It’s a bit of a resource hog and might not handle new, unseen attack types as well.

### 2. Spectrogram-based CNN
- **Innovation**: This one turns audio into 2D Mel-spectrograms—kind of like images—and lets a CNN pick out patterns that scream "deepfake."
- **Metrics**: It achieves an EER of 4.1-6.5% on the CD-ADD dataset, especially with models like Wav2Vec2 and Whisper.
- **Why I Like It**: It strikes a nice balance between accuracy and efficiency, making it a good fit for near-real-time detection. Plus, it handles a variety of AI-generated speech and real chats nicely.
- **Downsides**: You need to preprocess the audio into spectrograms, and how well it works depends a lot on the dataset’s variety.

### 3. End-to-End Waveform-based Model
- **Innovation**: This method skips the middleman and processes raw audio straight through with deep neural networks, like RawNet2.
- **Metrics**: It pulls off an EER of about 5-7% on the ASVspoof 2021 dataset.
- **Why I Like It**: It’s great for real-time applications since it doesn’t need much preprocessing and can tackle real conversations well.
- **Downsides**: It’s pretty demanding on resources and might overfit to the training data.

**My Pick**: I went with the spectrogram-based CNN because it felt manageable with the tools I had, like Librosa and TensorFlow, and it offered a solid middle ground compared to the heavier AASIST and the overfitting risks of the end-to-end model.

## Part 2: Implementation

### Implementation Comparison
- **AASIST**: Works with raw waveforms and attention, but it’s a bit too heavy on the system for my setup.
- **End-to-End Waveform**: Goes straight from audio to deep networks, but it could overfit with a smaller dataset and needs a lot of power.
- **Spectrogram CNN**: This is my choice—it uses preprocessed spectrograms with a CNN, which plays nice with tools I’m familiar with and works well with the CD-ADD data.

### Code and Dataset
- **Model**: I built a spectrogram-based CNN with 3 layers of convolution (32, 64, 128 filters), some MaxPooling, and Dense layers to figure out if it’s real or fake.
- **Dataset**: I used a piece of the CD-ADD dataset—`dataset_TED-LIUM.zip` (1.1 GB)—which I grabbed from https://openxlab.org.cn/datasets/ylaeo/CD-ADD/tree/main. It’s got both real and AI-generated speech to work with.
- **Notebook**: Check out `momenta_audio_deepfake.ipynb` for all the code and steps.

## Part 3: Documentation & Analysis

### Implementation Process
- **Challenges I Ran Into**:
  - At first, my `X` list was empty because I didn’t account for the nested folders in the dataset.
  - I wasn’t sure how to label things since some folders had different audio files (like missing `valle.wav` sometimes).
- **How I Fixed It**:
  - I switched to `os.walk` to dig through all the subfolders and grab every `.wav` file.
  - I decided `real.wav` was the genuine stuff (label 0) and everything else was a deepfake (label 1), then double-checked it made sense with the dataset.
- **Assumptions I Made**:
  - Any file that’s not `real.wav` is a deepfake, based on how CD-ADD seems to be set up.
  - The 1.1 GB `dataset_ted` chunk is good enough for a quick test run.

### Analysis
- **Why I Picked This**:
  - I went with the spectrogram-based CNN because it felt doable with the resources I had, using familiar libraries, and it didn’t overwhelm me like the other options might have.
- **How It Works**:
  - It takes audio, turns it into 128x128 Mel-spectrograms with Librosa, and lets a CNN—with its convolutional layers—spot deepfake patterns. Then it uses dense layers and a softmax to decide if it’s real or fake.
- **Performance Results**:
  - I got a test accuracy of 0.9937 and a test loss of 0.0188 after 10 epochs. The model learned fast and didn’t seem to overfit too much—check the plot for proof!
- **What I Liked**:
  - It hit a super high accuracy (99.37%) on my dataset slice and handled the preprocessing smoothly.
- **What Could Be Better**:
  - The small 1.1 GB dataset might mean it’s too tuned to what it’s seen, so there’s a risk of overfitting.
- **Ideas for the Future**:
  - Maybe add some data augmentation, like stretching the audio a bit with `librosa.effects.time_stretch`.
  - Bring in the full CD-ADD dataset (like `dataset_LibriTTS.zip`, 1.1 TB) for more variety.
  - Try a deeper model, like ResNet, to dig out more details.

### Reflection Questions
1. **Biggest Hurdles**:
   - Figuring out the nested folders and guessing the labels without a clear guide was tricky.
2. **Real-World vs. Research**:
   - It might not do as well out there with noisy audio, different accents, or new TTS tricks since my dataset is pretty limited.
3. **More Data or Tools?**:
   - I’d love the full CD-ADD, maybe ASVspoof 2021, or some real chat audio. A GPU would also speed things up.
4. **Getting It Live**:
   - I’d use TensorFlow Lite to run it on edge devices, buffer audio for real-time spectrograms, keep an eye on how it drifts, and plan to retrain it now and then.

## Setup Instructions
1. **Grab the Repo**:
git clone https://github.com/NOTMUKUNDVINAYAK/Momenta-assignment.git
cd momenta-assignment

2. **Get the Tools**:
pip install -r requirements.txt

3. **Download the Dataset**:
- Grab it from https://openxlab.org.cn/datasets/ylaeo/CD-ADD/tree/main.
- Download `dataset_TED-LIUM.zip` (1.1 GB) and unzip it to `D:\projects\Momenta project\dataset_ted` (tweak the path if yours is different).
- For the full dataset, consider `dataset_LibriTTS.zip` 

**Run the Notebook**:
jupyter notebook momenta_audio_deepfake.ipynb

- Run all the cells to see the training results and that plot!

## Dependencies
- Check out `requirements.txt` for the full list of packages.
