# <span style="color: #d22c19;">Première Séance</span>


# Architecture Decision Record — Cas 1 : Analyse d'images dermatologiques

## Choix du cas d'usage et compatibilité avec le quota

Le cas d'usage retenu est l'analyse d'images dermatologiques (Cas 1), 
qui impose un quota de 3500m CPU et 5 Gi de mémoire. 
Ce cas repose sur le dataset HAM10000, composé de 10 015 images JPEG de 
lésions cutanées réparties en 7 classes diagnostiques. 
Le dataset est disponible sur Harvard Dataverse (doi:10.7910/DVN/DBW86T) 
ainsi que sur Kaggle et l'archive ISIC, sous licence CC BY-NC 4.0, 
ce qui autorise un usage académique non commercial.

L'architecture retenue repose sur le modèle MobileNetV2 pré-entraîné sur ImageNet, 
fine-tuné pour les deux tâches de classification. MobileNetV2 a été préféré à 
EfficientNet-B0 pour sa légèreté en mémoire : ses poids représentent 
environ 14 Mo sur disque, contre 20 Mo pour EfficientNet-B0. 
En mémoire d'exécution sous PyTorch, un modèle MobileNetV2 consomme environ 50 à 80 Mo
une fois chargé, auxquels s'ajoute le runtime PyTorch d'environ 400 à 600 Mo. 
Deux instances de MobileNetV2 dans un même processus partagent ce runtime, 
portant la consommation totale du service d'inférence à environ 700 Mo à 1 Go. 
En prévoyant une marge pour les tenseurs intermédiaires et les pics d'inférence 
concurrente, une allocation de 2 Gi pour le service d'inférence est raisonnable et sûre.

Le service de preprocessing reçoit les images JPEG brutes, les redimensionne à 224×224 
pixels et les normalise avant transmission au service d'inférence. 
Une image JPEG de 800 Ko représente environ 600 Ko à 1 Mo en mémoire décompressée. 
Sous charge nominale de 50 requêtes par minute, le service peut traiter jusqu'à 5 à 10 images
simultanément, soit environ 10 Mo de données images en vol. En ajoutant le runtime Python
et les dépendances (Pillow, Flask), une allocation de 512 Mi couvre largement ce besoin. 
Le service de monitoring, qui se limite à compter les requêtes, enregistrer les latences
 et exposer des métriques via un endpoint REST, ne nécessite que 256 Mi.

Le budget mémoire total est donc de:
2 Gi (inférence) + 512 Mi (preprocessing) + 256 Mi (monitoring) = 2816 Mi, 
soit 2,75 Gi sur un quota de 5 Gi. Cette marge de 2,25 Gi permet d'absorber
les pics de consommation sous charge et d'envisager un troisième modèle Grad-CAM 
optionnel sans dépasser le quota. Côté CPU, l'inférence MobileNetV2 est l'opération 
la plus coûteuse ; une allocation de 1500m pour l'inférence, 1000m pour le preprocessing 
et 500m pour le monitoring totalise 3000m sur 3500m, laissant une marge de 500m.

## Communication entre services et placement du deuxième modèle

Les trois services communiquent via des appels REST internes au cluster Kubernetes. 
Le preprocessing reçoit la requête externe sur POST /predict, transforme l'image, 
puis transmet la requête au service d'inférence via un appel HTTP interne. 
Le service d'inférence exécute d'abord le classifieur binaire (bénin/malin). 
Si la prédiction est « malin » avec une confiance supérieure à un seuil configurable 
(par défaut 0.5), le classifieur multi-classes à 7 catégories est invoqué dans le même 
processus. Le deuxième modèle est donc hébergé dans le même service d'inférence que le 
premier. Ce choix est motivé par le partage du runtime PyTorch, qui représente la majeure 
partie de la consommation mémoire : séparer les modèles dans deux services distincts 
doublerait ce coût fixe, passant de 700 Mo à 1,4 Go de runtime seul, ce qui réduirait 
significativement la marge disponible sous le quota de 5 Gi.

Le service de monitoring interroge périodiquement les services via leurs endpoints de 
métriques ou reçoit les données en push. Il expose un endpoint GET /metrics qui rend 
lisibles le volume de requêtes, la latence moyenne, la latence P95 et le taux d'erreur.

## Choix CI/CD

L'outil CI/CD retenu est GitHub Actions, pour deux raisons concrètes. Premièrement, 
l'intégration native avec le dépôt GitHub élimine toute configuration d'infrastructure 
externe (pas de serveur Jenkins, pas de runner GitLab à provisionner). Deuxièmement, 
le plan gratuit offre 2 000 minutes par mois sur runners Ubuntu, ce qui est amplement 
suffisant pour un pipeline de tests et de build d'images Docker dans le cadre de ce projet. 
Le pipeline exécute les tests unitaires avec un seuil de couverture de 80 %, construit 
les trois images Docker et les pousse sur Docker Hub avec un tag de version. 
Le pipeline échoue si les tests échouent.

## Stratégie de déploiement

La stratégie de déploiement retenue est Recreate. Avec RollingUpdate, Kubernetes 
maintient temporairement l'ancien et le nouveau pod en parallèle pendant la 
transition (surge). Pour le service d'inférence avec une request de 2 Gi de mémoire, 
un surge de même valeur porterait la consommation temporaire à 4 Gi pour ce seul service, 
dépassant le budget restant après les autres services. Le calcul explicite est le suivant : 
en RollingUpdate avec maxSurge=1, le pic temporaire serait de: 
2 Gi × 2 (inférence) + 512 Mi (preprocessing) + 256 Mi (monitoring) = 4,75 Gi, 
ce qui ne laisse que 256 Mi de marge dans un quota de 5 Gi — insuffisant 
pour absorber un quelconque pic. En Recreate, les anciens pods sont supprimés 
avant la création des nouveaux, ce qui garantit que la consommation ne dépasse 
jamais le budget nominal de 2,75 Gi. Le compromis est une interruption de service brève
 pendant le redéploiement, acceptable dans un notre contexte.


# <span style="color: #d22c19;">Deuxieme Séance</span>
