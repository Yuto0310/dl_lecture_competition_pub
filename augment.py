import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import json

# データ拡張の定義
augmentations = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

# Resizeの定義
resize_transform = transforms.Resize((224, 224))

def augment_and_save(df_path, image_dir, save_dir, num_augmentations):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_json(df_path, orient='columns')
    new_data = []

    # オリジナルデータの処理
    for i, index in enumerate(df.index):
        row = df.loc[index]
        image_path = os.path.join(image_dir, row['image'])
        image = Image.open(image_path)

        # 元の画像をリサイズして保存
        resized_image = resize_transform(image)
        original_image_path = os.path.join(save_dir, f'{i}_original.png')
        resized_image.save(original_image_path)

        # 更新されたデータの保存
        new_row = {
            'image': f'{i}_original.png',
            'question': row['question'],
            'answers': row.get('answers', None)
        }
        new_data.append(new_row)

        # 拡張画像を保存
        for j in range(num_augmentations):
            augmented_image = augmentations(image)
            augmented_image = transforms.ToPILImage()(augmented_image)
            augmented_image_index = f"{i}_{j}"
            augmented_image_path = os.path.join(save_dir, f'{augmented_image_index}.png')
            augmented_image.save(augmented_image_path)

            # 更新されたデータの保存
            new_row = {
                'image': f'{augmented_image_index}.png',
                'question': row['question'],
                'answers': row.get('answers', None)
            }
            new_data.append(new_row)

    # 辞書形式のデータをDataFrameに変換して保存
    augmented_df = pd.DataFrame(new_data)
    augmented_df.to_json(os.path.join(save_dir, 'augmented_train.json'), orient='columns')

if __name__ == "__main__":
    augment_and_save(df_path="./data/train.json", image_dir="./data/train", save_dir="./data/augmented_train", num_augmentations=2)
