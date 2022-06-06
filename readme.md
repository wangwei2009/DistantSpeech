## DistantSpeech

**A python package for TRUE real-time speech enhancement**

- Install

```
pip install -e .
```

- Usage

```
cd DistantSpeech
source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
swig -python DistantSpeech/noise_estimation/mcra.i
gcc -c DistantSpeech/noise_estimation/mcra.c DistantSpeech/noise_estimation/mcra_wrap.c -I/home/wangwei/work/anaconda3/include/python3.7m -fPIC
```

  

