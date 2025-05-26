<h1 align="center">Camshow Deepfaker</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FCamshow Deepfaker | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

## About

This deepfake software is designed for the AI-generated media industry. It can assist in animating custom characters, creating engaging content, and using models for clothing design.

This software is part of a project to help redistribute income from NSFW content to camshow models in developing nations who have fewer job opportunities and lower incomes. By providing this tool, we aim to help these individuals pursue their long-term goals such as education.

Users are responsible for ensuring their use of the software complies with applicable laws and regulations.

## Quick Start

This open-source project can be installed manually following the instructions below.

## TLDR; Live Deepfake in just 3 Clicks
![easysteps](https://github.com/user-attachments/assets/af825228-852c-411b-b787-ffd9aac72fc6)
1. Select a face
2. Select which camera to use
3. Press live!

## Features & Uses - Everything is in real-time

### Mouth Mask

**Retain your original mouth for accurate movement using Mouth Mask**

<p align="center">
  <img src="media/ludwig.gif" alt="resizable-gif">
</p>

### Face Mapping

**Use different faces on multiple subjects simultaneously**

<p align="center">
  <img src="media/streamers.gif" alt="face_mapping_source">
</p>

### Your Movie, Your Face

**Watch movies with any face in real-time**

<p align="center">
  <img src="media/movie.gif" alt="movie">
</p>

### Live Show

**Run Live shows and performances**

<p align="center">
  <img src="media/live_show.gif" alt="show">
</p>

### Memes

**Create Your Most Viral Meme Yet**

<p align="center">
  <img src="media/meme.gif" alt="show" width="450"> 
  <br>
  <sub>Created using Many Faces feature in Camshow Deepfaker</sub>
</p>

### Omegle

**Surprise people on Omegle**

<p align="center">
  <video src="https://github.com/user-attachments/assets/2e9b9b82-fa04-4b70-9f56-b1f68e7672d0" width="450" controls></video>
</p>

## Installation (Manual)

**Please be aware that the installation requires technical skills and is not for beginners. Consider downloading the prebuilt version.**

<details>
<summary>Click to see the process</summary>

### Installation

This is more likely to work on your computer but will be slower as it utilizes the CPU.

**1. Set up Your Platform**

-   Python 3.10 (required)
-   pip
-   git
-   [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA) - ```iex (irm ffmpeg.tc.ht)```

**2. Clone the Repository**

```bash
git clone https://github.com/spectrumwebco/camshow-deepfaker.git
cd camshow-deepfaker
```

**3. Download the Models**

1. [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
2. [inswapper\_128\_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Place these files in the "**models**" folder.

**4. Install Dependencies with UV**

We use UV for fast, reliable dependency management. UV is powered by Rust for superior performance.

**Installation Script (Recommended):**
```bash
chmod +x install.sh
./install.sh
```

**Manual Installation:**

For **Linux (Ubuntu 24.04 LTS)**:
```bash
# Install UV
pip install uv

# Install with CUDA support (if NVIDIA GPU detected)
uv pip install -e ".[linux,dev]"

# Or CPU-only version
uv pip install -e ".[dev]"
```

For **macOS (Apple Silicon M1/M2/M3)**:
```bash
# Install Python 3.10 (required)
brew install python@3.10
brew install python-tk@3.10

# Install UV
pip install uv

# Install with Apple Silicon optimizations
uv pip install -e ".[macos-silicon,dev]"
```

For **macOS (Intel)**:
```bash
# Install UV
pip install uv

# Install dependencies
uv pip install -e ".[macos,dev]"
```

**In case something goes wrong and you need to reinstall:**

```bash
# Install UV if not already installed
pip install uv

# Reinstall dependencies
uv pip sync
```

**5. Run the Application**

**GUI Mode (Traditional):**
```bash
python run.py
```

**API Mode (Headless/Production):**
```bash
python app.py
# Or use the platform-optimized script
./run_api.sh
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### GPU Acceleration

**CUDA Execution Provider (Nvidia)**

1. Install [CUDA Toolkit 11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```

3. Usage:

```bash
python run.py --execution-provider cuda
```

**CoreML Execution Provider (Apple Silicon)**

Apple Silicon (M1/M2/M3) specific installation:

1. Make sure you've completed the macOS setup above using Python 3.10.
2. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-silicon
pip install onnxruntime-silicon==1.13.1
```

3. Usage (important: specify Python 3.10):

```bash
python3.10 run.py --execution-provider coreml
```

**Important Notes for macOS:**
- You **must** use Python 3.10, not newer versions like 3.11 or 3.13
- Always run with `python3.10` command not just `python` if you have multiple Python versions installed
- If you get error about `_tkinter` missing, reinstall the tkinter package: `brew reinstall python-tk@3.10`
- If you get model loading errors, check that your models are in the correct folder
- If you encounter conflicts with other Python versions, consider uninstalling them:
  ```bash
  # List all installed Python versions
  brew list | grep python
  
  # Uninstall conflicting versions if needed
  brew uninstall --ignore-dependencies python@3.11 python@3.13
  
  # Keep only Python 3.10
  brew cleanup
  ```

**CoreML Execution Provider (Apple Legacy)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-coreml
pip install onnxruntime-coreml==1.13.1
```

2. Usage:

```bash
python run.py --execution-provider coreml
```



**OpenVINO™ Execution Provider (Intel)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino==1.15.0
```

2. Usage:

```bash
python run.py --execution-provider openvino
```
</details>

## Usage

**1. Image/Video Mode (GUI)**

-   Execute `python run.py`.
-   Choose a source face image and a target image/video.
-   Click "Start".
-   The output will be saved in a directory named after the target video.

**2. Webcam Mode (GUI)**

-   Execute `python run.py`.
-   Select a source face image.
-   Click "Live".
-   Wait for the preview to appear (10-30 seconds).
-   Use a screen capture tool like OBS to stream.
-   To change the face, select a new source image.

**3. API Mode (Headless)**

-   Execute `python app.py` or `./run_api.sh`.
-   The API server will start on port 8000.
-   Access the API documentation at `http://localhost:8000/docs`.
-   Use the API endpoints to upload files, process media, and download results.

## API Documentation

The Camshow Deepfaker API provides a headless interface for face swapping and deepfake functionality, ideal for server deployments and integration with other applications.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/upload` | POST | Upload source or target files |
| `/process` | POST | Process media with face swap |
| `/download/{filename}` | GET | Download processed files |
| `/status` | GET | Get system status |

### Example Usage

**Upload a file:**
```bash
curl -X POST -F "file=@/path/to/source.jpg" http://localhost:8000/upload
```

**Process media:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"source_path": "uploads/source.jpg", "target_path": "uploads/target.mp4"}' \
  http://localhost:8000/process
```

**Download processed file:**
```bash
curl -X GET http://localhost:8000/download/target.mp4 -o processed_video.mp4
```

### Integration with PyO3 Rust Modules

The API is designed to work seamlessly with future PyO3 Rust modules. As performance-critical components are migrated to Rust, they will be integrated with the FastAPI backend without changing the API interface.

## Tips and Tricks

Check out these helpful guides to get the most out of Camshow Deepfaker:

- [Unlocking the Secrets to the Perfect Deepfake Image](https://camshowdeepfaker.com/index.php/blog/tips-and-tricks/unlocking-the-secrets-to-the-perfect-deepfake-image) - Learn how to create the best deepfake with full head coverage
- [Video Call with DeepLiveCam](https://camshowdeepfaker.com/index.php/blog/tips-and-tricks/video-call-with-deeplivecam) - Make your meetings livelier by using DeepLiveCam with OBS and meeting software
- [Have a Special Guest!](https://camshowdeepfaker.com/index.php/blog/tips-and-tricks/have-a-special-guest) - Tutorial on how to use face mapping to add special guests to your stream
- [Watch Deepfake Movies in Realtime](https://camshowdeepfaker.com/index.php/blog/tips-and-tricks/watch-deepfake-movies-in-realtime) - See yourself star in any video without processing the video
- [Better Quality without Sacrificing Speed](https://camshowdeepfaker.com/index.php/blog/tips-and-tricks/better-quality-without-sacrificing-speed) - Tips for achieving better results without impacting performance
- [Instant Vtuber!](https://camshowdeepfaker.com/index.php/blog/tips-and-tricks/instant-vtuber) - Create a new persona/vtuber easily using Metahuman Creator

Visit our [official blog](https://camshowdeepfaker.com/index.php/blog/tips-and-tricks) for more tips and tutorials.

## Command Line Arguments (Unmaintained)

```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select a source image
  -t TARGET_PATH, --target TARGET_PATH                     select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  frame processors (choices: face_swapper, face_enhancer, ...)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --map-faces                                              map source target faces
  --mouth-mask                                             mask the mouth region
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  --live-mirror                                            the live camera display as you see it in the front-facing camera frame
  --live-resizable                                         the live camera frame is resizable
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu} [{cpu} ...]                   available execution provider (choices: cpu, ...)
  --execution-threads EXECUTION_THREADS                    number of execution threads
  -v, --version                                            show program's version number and exit
```

Looking for a CLI mode? Using the -s/--source argument will make the run program in cli mode.

## Press

**We are always open to criticism and are ready to improve, that's why we didn't cherry-pick anything.**

 - [*"Camshow Deepfaker goes viral, allowing anyone to become a digital doppelganger"*](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) - Ars Technica
 - [*"Thanks Deep Live Cam, shapeshifters are among us now"*](https://dataconomy.com/2024/08/15/what-is-deep-live-cam-github-deepfake/) - Dataconomy
 - [*"This free AI tool lets you become anyone during video-calls"*](https://www.newsbytesapp.com/news/science/deep-live-cam-ai-impersonation-tool-goes-viral/story) - NewsBytes
 - [*"OK, this viral AI live stream software is truly terrifying"*](https://www.creativebloq.com/ai/ok-this-viral-ai-live-stream-software-is-truly-terrifying) - Creative Bloq
 - [*"Deepfake AI Tool Lets You Become Anyone in a Video Call With Single Photo"*](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/) - PetaPixel
 - [*"Camshow Deepfaker Uses AI to Transform Your Face in Real-Time, Celebrities Included"*](https://www.techeblog.com/deep-live-cam-ai-transform-face/) - TechEBlog
 - [*"An AI tool that "makes you look like anyone" during a video call is going viral online"*](https://telegrafi.com/en/a-tool-that-makes-you-look-like-anyone-during-a-video-call-is-going-viral-on-the-Internet/) - Telegrafi
 - [*"This Deepfake Tool Turning Images Into Livestreams is Topping the GitHub Charts"*](https://decrypt.co/244565/this-deepfake-tool-turning-images-into-livestreams-is-topping-the-github-charts) - Emerge
 - [*"New Real-Time Face-Swapping AI Allows Anyone to Mimic Famous Faces"*](https://www.digitalmusicnews.com/2024/08/15/face-swapping-ai-real-time-mimic/) - Digital Music News
 - [*"This real-time webcam deepfake tool raises alarms about the future of identity theft"*](https://www.diyphotography.net/this-real-time-webcam-deepfake-tool-raises-alarms-about-the-future-of-identity-theft/) - DIYPhotography
 - [*"That's Crazy, Oh God. That's Fucking Freaky Dude... That's So Wild Dude"*](https://www.youtube.com/watch?time_continue=1074&v=py4Tc-Y8BcY) - SomeOrdinaryGamers
 - [*"Alright look look look, now look chat, we can do any face we want to look like chat"*](https://www.youtube.com/live/mFsCe7AIxq8?feature=shared&t=2686) - IShowSpeed

## Credits

-   [ffmpeg](https://ffmpeg.org/): for making video-related operations easy
-   [deepinsight](https://github.com/deepinsight): for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models. Please be reminded that the [use of the model is for non-commercial research purposes only](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
-   [havok2-htwo](https://github.com/havok2-htwo): for sharing the code for webcam
-   [GosuDRM](https://github.com/GosuDRM): for the open version of roop
-   [pereiraroland26](https://github.com/pereiraroland26): Multiple faces support
-   [vic4key](https://github.com/vic4key): For supporting/contributing to this project
-   [kier007](https://github.com/kier007): for improving the user experience
-   [qitianai](https://github.com/qitianai): for multi-lingual support
-   and [all developers](https://github.com/hacksider/Camshow Deepfaker/graphs/contributors) behind libraries used in this project.
-   Footnote: Please be informed that the base author of the code is [s0md3v](https://github.com/s0md3v/roop)
-   All the wonderful users who helped make this project go viral by starring the repo ❤️

[![Stargazers](https://reporoster.com/stars/spectrumwebco/camshow-deepfaker)](https://github.com/spectrumwebco/camshow-deepfaker/stargazers)

## Contributions

![Alt](https://repobeats.axiom.co/api/embed/fec8e29c45dfdb9c5916f3a7830e1249308d20e1.svg "Repobeats analytics image")

## Stars to the Moon 🚀

<a href="https://star-history.com/#spectrumwebco/camshow-deepfaker&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=spectrumwebco/camshow-deepfaker&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=spectrumwebco/camshow-deepfaker&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=spectrumwebco/camshow-deepfaker&type=Date" />
 </picture>
</a>
