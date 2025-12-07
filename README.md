# AA228Roomba

## Installation

Install required Python packages into your virtual environment:

```
pip install -r requirements.txt
```

Notes:
- This project now prefers `gymnasium`. The `requirements.txt` includes `gymnasium` and keeps `gym` for backward compatibility.
- If you plan to use the recording/video helpers (`save_frame` / `make_video`), install the video backend plugin for `imageio` (the requirements include `imageio[ffmpeg]`). On Windows you can also install `imageio[pyav]` if preferred.

Example additional setup (if needed):

```
pip install imageio[ffmpeg]
# or
pip install imageio[pyav]
```

When running demos that create MP4s, ensure `ffmpeg` is available (the `imageio[ffmpeg]` wheel will download a bundled ffmpeg on install). If you prefer a system `ffmpeg`, you can install it separately and make sure it's on your PATH.



## Running Sarsa/Qlearning

Run the command:
python main.py --method sarsa --scenario single_soft_center --episodes 500 --log-transitions

--method takes in 'sarsa' or 'qlearning'
--scenario takes any of the pre-defined scenarios in /test_scenarios
--log-transitions logs the result into csv files

## Running plots
python compare_methods.py
Hardcode csv file names in the python file so adjust as needed
