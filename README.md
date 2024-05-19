# gpt-object-detection

## requirements
- Python 3.12.2
- poetry

## install
```bash
poetry install
```

## run
```bash
poetry run python main.py --image <image_path> --labels <label_list> [--use_dot_matrix true]
```

### example
```bash
poetry run python main.py --image target.jpg --labels person, bike --use_dot_matrix true
```

