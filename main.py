import argparse
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from openai import OpenAI

load_dotenv(dotenv_path=".env")
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--use_dot_matrix", type=bool, default=False)
    parser.add_argument("--preview", type=bool, default=False)
    parser.add_argument("--show_chat", type=bool, default=False)
    return parser.parse_args()


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def overlay_dot_matrix(image_path: Path) -> Path:
    STEP = 10
    image = cv2.imread(str(image_path))
    height, width, _ = image.shape

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    step_x = width // STEP
    step_y = height // STEP

    for i in range(1, STEP):
        for j in range(1, STEP):
            x = i * step_x
            y = j * step_y

            if binary_image[y, x] == 0:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)

            cv2.circle(image, (x, y), 3, color, -1)

            cv2.putText(
                image,
                f"({i / STEP:.1f},{j / STEP:.1f})",
                (x - 20, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

    output_path = image_path.parent / (image_path.stem + "_with_dot_matrix" + image_path.suffix)
    cv2.imwrite(str(output_path), image)
    return output_path


def assign_label_colors(labels: list[str]) -> dict[str, tuple[int, int, int]]:
    colors = {}
    for x in labels:
        if x not in colors:
            colors[x] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    return colors


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")
    collect_annotation_path = Path("collect_annotation.json")
    if not collect_annotation_path.exists():
        raise FileNotFoundError(f"File not found: {collect_annotation_path}")
    labels = args.labels
    is_use_dot_matrix = args.use_dot_matrix
    is_preview = args.preview
    is_show_chat = args.show_chat

    original_image = cv2.imread(str(image_path))
    height, width, _ = original_image.shape

    if is_use_dot_matrix:
        image_path = overlay_dot_matrix(image_path)
        additional_prompt = """
            画像中には、その位置の目印として、ドットとその座標が(x座標, y座標)の形式で記入されています。
            ドットは9x9の81個存在し、左上のドットが(0.1, 0.1)、右下のドットが(0.9, 0.9)に位置しています。
            また、画像の左上端が(0.0, 0.0)、右下端が(1.0, 1.0)です。
            あなたはこの情報を利用して物体の位置を特定してください。以下の手順を参考にしてください：
            1.	検出対象の物体を認識し、どのように配置されているかを説明する
            2.	認識した物体のの上下左右それぞれの端に最も近いドットの位置を特定し、その座標を説明する
            3.	ドットの座標を用いて、物体の境界を示す矩形の位置を特定する。複数のドットがある場合、平均値を取るなどして精度を上げる。
        """
    else:
        additional_prompt = ""

    base64_image = encode_image(image_path)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = f"""
    あなたは画像中の物体を検出する役割を担っています。
    userから与えられる画像の中から、{', '.join(labels)}に該当するすべての物体を検出し、その物体を過不足なく抽出することができる矩形の座標情報を教えてください。
    あなたが出力する矩形は、物体の全てを完全に囲み、またできる限り物体にフィットしている必要があります。
    {additional_prompt}
    検出結果は、次のようなJSON形式で出力する必要があります。座標値は画像の左上を(0.0, 0.0)、右下を(1.0, 1.0)とした相対座標で表現してください。
    {{"num_annotations": 2, "annotations": [{{"label": "label1", "coordinates": {{"top": 0.15, "right": 0.25, "bottom": 0.35, "left": 0.40}}}}, {{"label": "label2", "coordinates": {{"top": 0.50, "right": 0.65, "bottom": 0.75, "left": 0.80}}}}]}}
    ここで、num_annotationsは検出した物体の数、labelは{', '.join(labels)}のうちどのラベルに該当するかを、coordinatesは矩形の座標情報をそれぞれ表しています。
    この作業を2つのSTEPに分けて実施します。STEP1では、検出対象の物体の左上と右下の位置を可能な限り正確に特定します。STEP2では、STEP1で特定した位置をもとに、物体のラベルを特定し、その情報をJSON形式で返してください。
    """
    user_prompt = """
        この画像に写っているものを検出して下さい。
        まずSTEP1を始めて下さい。
    """

    contents = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
                },
            ],
        },
    ]

    # STEP1
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=contents,
        max_tokens=300,
    )

    answer = response.choices[0].message.content
    if answer is None:
        raise ValueError("STEP1: answer is None")
    if is_show_chat:
        print(answer)

    # STEP2
    contents.append({"role": "assistant", "content": answer})
    contents.append({"role": "user", "content": "STEP2を始めて下さい"})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=contents,
        response_format={"type": "json_object"},
        max_tokens=300,
    )

    annotation_json = response.choices[0].message.content
    if is_show_chat:
        print(annotation_json)

    if annotation_json is not None:
        try:
            annotation_dict = json.loads(annotation_json)
            annotations = annotation_dict["annotations"]
        except json.JSONDecodeError:
            raise ValueError("STEP2: answer is not valid JSON")

        label_colors = assign_label_colors(labels)
        for x in annotations:
            try:
                label = x["label"]
                coordinates = x["coordinates"]
            except KeyError:
                raise ValueError("STEP2: answer is not valid JSON")

            if label in label_colors:
                color = label_colors[label]
            else:
                color = (0, 0, 0)

            top = int(coordinates["top"] * height)
            right = int(coordinates["right"] * width)
            bottom = int(coordinates["bottom"] * height)
            left = int(coordinates["left"] * width)

            cv2.rectangle(original_image, (left, top), (right, bottom), color, 2)
            cv2.putText(original_image, f"{x['label']}", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.savefig("result.jpg")
        if is_preview:
            plt.show()
    else:
        raise ValueError("STEP2: answer is None")


if __name__ == "__main__":
    main()
