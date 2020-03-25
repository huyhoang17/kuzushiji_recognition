[![](https://forthebadge.com/images/badges/built-by-developers.svg)](https://forthebadge.com)
[![](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/huyhoang17/kuzushiji_recognition/)
[![](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://forthebadge.com)

# Kuzushiji Recognition
[Late Submission] Solution for kuzushiji recognition (kaggle competition)

- Link blog post: [Building OCR module for Kuzushiji recognition](https://viblo.asia/p/V3m5WPngKO7)

### Segmentation model

- Unet with custom resnet-based backbone

##### Evaluate on detection model

![](./assets/eval_detection_result.png)

### Classification model

- Baseline model for kuzushiji character recognition

- Number of classes: 3422

### Command

- Clone repository

```python
git clone https://github.com/huyhoang17/kuzushiji_recognition
cd kuzushiji_recognition
```

- Install some prerequisite libs

```python
pip install -r requirements.txt
```

- [Optional] Install `Git LFS` and pull model files, follow by this [tutorial](https://www.atlassian.com/git/tutorials/git-lfs#pulling-and-checking-out)

- Open kuzu_tfserving.config on editor, change `base_path` of 2 models to absolute path to each sub-folder

```bash
# change this line
base_path: '/home/phan.huy.hoang/workspace/projects/kaggle_kuzushiji/model_server/kuzu_segment'

# to
base_path: '/absolute-path-to-root-folder/model_server/kuzu_segment'
```

- Install `tensorflow_model_server` from this [link](https://www.tensorflow.org/tfx/serving/setup)

- Run tensorflow model server

```bash
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=/absolute-path-to-kuzu-tfserving.config
```

- Test detection & recognition model

```bash
python3 src/grpc_infer.py
```

- Check result image in `assets` folder

### Result

![](./assets/result1.jpg)

![](./assets/result2.jpg)

### TODO

- Add pytorch code

- Update docker / docker-compose

### Contact

- If you find this repo useful, please star the project to let people know that it's reliable :star::star::star: Thank you!

- For more information, please contact me at email address: hoangphan0710@gmail.com
