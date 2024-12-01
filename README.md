# Environnement Python et Dépendances (avec Anaconda)

Pour garantir la reproductibilité des résultats, suivez ce guide pour configurer l’environnement Python et installer les dépendances nécessaires.

## 1. Version de Python
Le programme est testé avec **Python 3.8** ou version ultérieure. Utilisez cette version pour éviter des problèmes de compatibilité.

---

## 2. Configuration de l’Environnement Virtuel (avec Anaconda)

### Étape 1 : Installer Anaconda
Téléchargez et installez [Anaconda](https://www.anaconda.com/).

### Étape 2 : Créer un nouvel environnement virtuel
Dans l'Anaconda Prompt ou le terminal, exécutez :
```bash
conda create --name tsr_env python=3.8
```
### Étape 3 : Activer l’environnement virtuel
Exécutez :
```bash
conda activate tsr_env
```
### Étape 4 : Vérifier l’activation de l’environnement
Exécutez :
```bash
conda info --envs
```
L’environnement actif sera marqué par un astérisque *

## 3. Bibliothèques Nécessaires
Liste des bibliothèques
| **Bibliothèque**   | **Version (testée)** | **Objectif**                               |
|--------------------|----------------------|-------------------------------------------|
| `numpy`           | >= 1.19.5           | Calculs numériques                        |
| `pandas`          | >= 1.1.5            | Manipulation de données                   |
| `opencv-python`   | >= 4.5.1            | Traitement d’image                        |
| `matplotlib`      | >= 3.3.4            | Visualisation                             |
| `tensorflow`      | >= 2.4.1            | Réseaux neuronaux                         |
| `scikit-learn`    | >= 0.24.1           | Modèles et métriques                      |
| `Pillow`          | >= 8.1.0            | Manipulation d’images                     |

Installation des bibliothèques
Avec pip :
```bash
pip install numpy pandas opencv-python matplotlib tensorflow scikit-learn Pillow
```

Avec conda :
```bash
conda install numpy pandas opencv matplotlib tensorflow scikit-learn pillow
```

## 4. Validation de l’Environnement
Testez la configuration avec :
```bash
python -c "import numpy, pandas, cv2, matplotlib, tensorflow, sklearn, PIL; print('L’environnement est correctement configuré.')"
```

## 5. Exigences Matérielles
+ Système d’exploitation : Windows, macOS ou Linux
+ RAM : 8GB minimum (16GB recommandé)
+ GPU (optionnel) : Pour accélérer l’entraînement, utilisez un GPU NVIDIA avec CUDA et cuDNN.
++ Installation pour TensorFlow GPU :
```bash
  pip install tensorflow-gpu
```

## 6. Exécution du Programme
Après ce qui précède, nous pouvons relancer l’expérience en exécutant le fichier main1.ipynb avec jupter notebook. Pour plus de détails ou de dépannage, consultez le fichier readme.md dans le dépôt GitHub.

### Télécharger et installer Jupyter Notebook

1. **Via pip** (recommandé) :  
   Si vous avez Python installé, exécutez la commande suivante :  
   ```bash
   pip install notebook
    ```
2. **Via conda** (avec Anaconda) :
  Si vous utilisez Anaconda, installez-le avec :
  ```bash
  conda install -c anaconda notebook
  ```
3. **Lancer Jupyter Notebook** :
Après l'installation, lancez Jupyter Notebook avec :
  ```bash
  jupyter notebook
  ```
## 7. Utilisation de TensorBoard

TensorBoard est un outil puissant pour visualiser les performances de vos modèles.

---

### 1. **Installation**
Installez TensorBoard via `pip` :
```bash
pip install tensorboard
```

### 2. **Ajout de TensorBoard dans votre code**
Ajoutez le callback dans votre script TensorFlow :
```python
import tensorflow as tf

# Répertoire de logs
log_dir = "logs/fit/"  

# Callback TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Entraînement avec le callback
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])

```

### 3. **Lancement de TensorBoard**
Lancez TensorBoard pour visualiser les logs :
```bash
tensorboard --logdir=logs/fit
```

### 4. **Accès à l'interface**
Une URL sera affichée (ex. http://localhost:6006). Ouvrez-la dans votre navigateur.

### 5. Fonctionnalités principales

Scalars : Visualisation des métriques comme accuracy ou loss.
Graphs : Structure du modèle (graph computationnel).
Histograms : Analyse des poids et activations.
Images : Visualisation d’images générées ou utilisées.








