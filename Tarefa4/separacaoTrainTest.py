import os
import shutil
import random

# Diretório raiz contendo as pastas de classes
dataset_root = 'frequenciasDefinitivas'

# Diretório de saída para o conjunto de treinamento
train_dir = 'train1'

# Diretório de saída para o conjunto de teste
test_dir = 'test1'

# Proporção de imagens a serem usadas para treinamento (por exemplo, 80%)
train_ratio = 0.8

# Lista de pastas (rótulos) no diretório raiz
labels = os.listdir(dataset_root)

# Loop através das pastas (rótulos)
for label in labels:
    label_dir = os.path.join(dataset_root, label)
    images = os.listdir(label_dir)
    random.shuffle(images)

    # Determina a quantidade de imagens para o treinamento e teste
    num_train = int(train_ratio * len(images))
    
    # Cria os diretórios de treinamento e teste
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    # Copia as imagens para os diretórios de treinamento e teste
    for i, image in enumerate(images):
        if i < num_train:
            dest_dir = os.path.join(train_dir, label)
        else:
            dest_dir = os.path.join(test_dir, label)
        shutil.copy(os.path.join(label_dir, image), os.path.join(dest_dir, image))

print("Divisão de treinamento e teste concluída.")
