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



# <span style="color: #d22c19;">Deuxième Séance : Conteneurisation, Optimisation et Pipeline CI/CD</span>
Optimisation des images sous contrainte de taille
La contrainte majeure de cette séance a résidé dans le respect strict de la taille des images conteneurisées et la gestion de la mémoire sous charge concurrente. Lors des premiers essais, l'image du service d'inférence atteignait une taille critique de 5,6 Go sur Docker Hub, un volume incompatible avec un déploiement Kubernetes fluide et provoquant des échecs de téléchargement systématiques par dépassement du délai d'attente de Minikube.
L'analyse des couches d'image a révélé que l'installation standard de PyTorch via le gestionnaire de paquets embarquait par défaut l'intégralité des binaires d'accélération Nvidia CUDA et cuDNN, ajoutant plus de 4 Go de pilotes inutiles pour une exécution CPU sur notre cluster local.
Pour corriger cette dérive sans altérer le code applicatif, deux optimisations ont été implémentées dans le processus de build de l'inférence :
	1.	Utilisation d'une image de base "Slim" : Remplacement de l'image Python standard par une version allégée, réduisant l'empreinte initiale du système d'exploitation de près de 700 Mo.
	2.	Forçage de l'index CPU : Isolation et ciblage explicite de la version CPU de PyTorch en amont du fichier de dépendances, court-circuitant ainsi les résolutions standards.
Grâce à ces mesures, le téléchargement de PyTorch a fondu et l'ensemble des dépendances CUDA a été banni. L'image finale poussée sur Docker Hub a atteint seulement 300 Mo, garantissant un démarrage des conteneurs en quelques secondes.
Robustesse du pipeline CI/CD Multi-Architecture
Le pipeline GitHub Actions a été configuré pour répondre à l'exigence multi-architecture, permettant la compilation simultanée pour les architectures serveurs standards et nativement pour la puce Apple Silicon de notre environnement de développement local.
Pour pallier le fait que les runners de GitHub Actions tournent exclusivement sur du matériel Intel, l'intégration de l'émulateur QEMU a été indispensable pour simuler l'architecture ARM64 en arrière-plan.
Afin d'éviter que le mécanisme de cache global de GitHub Actions ne conserve des traces des couches Nvidia CUDA lourdes initialement construites, le drapeau de désactivation du cache a été temporairement imposé lors de la phase de stabilisation. De plus, pour sanctuariser le contexte de build et interdire l'inclusion accidentelle d'environnements virtuels locaux ou de résidus de tests, un fichier d'exclusion strict a été positionné à la racine du projet.


# <span style="color: #d22c19;">Troisième Séance : Déploiement Kubernetes et Stabilisation du Cluster Local</span>
Initialisation et Réparation de l'infrastructure Minikube
Le déploiement sur l'environnement Minikube a mis en évidence des instabilités critiques liées à la corruption du conteneur de contrôle-nœud lors des phases d'arrêts brutaux, bloquant l'API Server et empêchant l'activation des extensions de stockage indispensables.
Pour retrouver un état d'infrastructure sain, reproductible et conforme aux spécifications matérielles recommandées, une purge complète du cluster local, des profils et des fichiers de configuration corrompus a été opérée avant de relancer proprement Minikube.
Résolution du verrou d'authentification Docker Hub
Suite à la réinitialisation du cluster, les pods ont immédiatement basculé en état d'erreur de téléchargement. L'analyse des événements du cycle de vie du pod d'inférence a mis en lumière un double blocage :
	1.	Une erreur indiquant que le cluster ne parvenait pas à localiser les clés d'accès au registre privé.
	2.	Une erreur de Docker Hub bloquant les requêtes anonymes par épuisement des quotas de téléchargement.
Pour résoudre ce problème de manière étanche au sein du namespace isolé, l'ancien secret défaillant a été supprimé puis recréé proprement avec des identifiants valides. Ce secret d'authentification a été explicitement rattaché aux directives des manifests Kubernetes, combiné à une politique de téléchargement systématique afin d'obliger le cluster à récupérer la version multi-architecture corrigée.
Intégration des Artefacts de Modèles au Runtime
Lors des phases de tests fonctionnels initiées par requêtes HTTP depuis le système hôte, le pipeline a renvoyé une erreur de service indisponible provoquée par la levée d'une exception fatale dans le code applicatif FastAPI. Bien que le processus prévoyait la création du dossier structurel pour les modèles, l'encapsulation de l'artefact entraîné hors Minikube de 20 Mo était manquante.
La correction définitive a consisté à modifier la topologie du build en ajoutant l'instruction de copie des artefacts de poids directement dans l'image lors de sa compilation.
Après un ultime redémarrage des déploiements pour forcer la prise en compte des modifications, le cluster s'est stabilisé. Tous les microservices (Preprocessing, Inference, Monitoring) sont désormais en statut Running. Le pont réseau valide la communication de bout en bout et confirme que le pipeline est prêt pour subir les stress tests de charge concurrentielle exigés par le correcteur.

# <span style="color: #d22c19;">Quatrième Séance : Gestion des Ressources sous Charge et Résolution des Goulots d'Étranglement</span>

## Comportement en Régime Nominal (10 req/min)
Lors du premier palier d'évaluation à 10 requêtes par minute, l'infrastructure s'est stabilisée de manière optimale, affichant un taux de succès de 100 % sur les 50 requêtes envoyées. Les relevés de consommation instantanés (`kubectl top pods`) ont indiqué les métriques au repos et en activité nominale suivantes :
* **Service d'Inférence** : Hausse de 232 Mi à 449 Mi de RAM lors de la première sollicitation (initialisation dynamique des tenseurs PyTorch et chargement des graphes de calcul du modèle MobileNetV2), puis stabilisation de la charge CPU à un niveau moyen de 107m (environ 10 % de la puissance réservée).
* **Service de Preprocessing** : Consommation de 9m à 11m CPU et 82 Mi de RAM, confirmant la légèreté des opérations de décodage et de redimensionnement des images à faible fréquence.
* **Service de Monitoring** : Charge minime de 2m CPU et 78 Mi de RAM dédiée à l'agrégation continue des métriques de latence.

Bien que la latence moyenne se soit établie à un excellent niveau (0,684s), un pic initial isolé à 5,485s a été mesuré, correspondant au démarrage à froid (*cold start*) de l'interpréteur et à l'allocation initiale de la mémoire matricielle par le framework de Deep Learning.

## Analyse des Défaillances sous Charge Concurrentielle (50 req/min)
Le passage au deuxième palier de charge (50 requêtes par minute, soit 250 requêtes au total) a mis en évidence une dégradation critique des performances, matérialisée par un taux d'échec de 19,6 % (49 requêtes perdues) et des latences (P95 et maximum) bloquées au plafond strict de 30,0 secondes.

Les analyses des métriques système réalisées toutes les 30 secondes ont révélé un comportement asymétrique du pipeline :
1. **L'Inférence sous contrainte** : Le CPU du pod d'inférence a subi une explosion instantanée, bondissant de 235m à 1075m CPU, tandis que son empreinte mémoire demeurait parfaitement stable à 447 Mi.
2. **Le Preprocessing au repos** : Le service en amont n'a consommé que 55m CPU (5,5 % de son allocation), excluant tout goulot d'étranglement lié à la manipulation des fichiers JPEG en mémoire.

Le diagnostic technique a confirmé que la limite physique imposée de 1500m CPU n'était pas atteinte par le conteneur, mais que l'architecture logicielle s'est heurtée au goulot d'étranglement du **mono-thréading de l'interpréteur Python**. Le serveur Uvicorn, configuré par défaut avec un unique processus ouvrier (*worker*), s'est retrouvé dans l'incapacité de traiter les calculs matriciels de MobileNetV2 de manière concurrente. Les requêtes se sont accumulées séquentiellement dans la file d'attente réseau jusqu'à dépasser le seuil de *timeout* de 30 secondes défini par le script de charge.

## Arbitrages Architecturaux et Optimisation In-Container
Pour remédier aux échecs de timeout sans enfreindre les règles strictes de quotas et de limites Kubernetes assignées au namespace (`projet-quota`), l'architecture interne des conteneurs a été révisée en appliquant deux correctifs majeurs :

* **Parallélisation Applicative Interne** : La configuration de démarrage du serveur de production au sein du Dockerfile d'inférence a été modifiée pour initier plusieurs processus ouvriers (`--workers 2`). Cette modification permet de saturer et d'exploiter efficacement les 425m de CPU disponibles restants sous la limite des 1500m alloués au conteneur, permettant à l'API de distribuer le calcul des prédictions sur plusieurs threads sans requérir de ressources Kubernetes supplémentaires.
* **Redistribution Alignée des Ressources (ADR / LimitRange)** : Afin d'offrir une marge de calcul supplémentaire au traitement de deep learning sans modifier l'enveloppe globale du ResourceQuota ($3500\text{m}$ CPU et $5\text{Gi}$ RAM), une redistribution des enveloppes a été opérée au sein du manifest `pipeline-ml.yaml`. La limite CPU de l'inférence a été portée à $2000\text{m}$ (le maximum absolu autorisé par le `LimitRange`), financée par une réduction de la réserve du monitoring et du preprocessing, optimisant ainsi l'usage de chaque cœur CPU disponible sur le cluster Minikube.

## Validation de la Correction et Comportement Post-Optimisation
À la suite de l'implémentation des 2 workers applicatifs et de l'alignement de la limite CPU à 2000m (conforme au plafond du LimitRange), le test de charge à 50 requêtes par minute a été réexécuté afin de mesurer l'impact réel de l'optimisation.

Les relevés séquentiels de consommation ont démontré une parfaite efficacité de la parallélisation interne :
* **Montée en charge CPU dynamique** : Le conteneur d'inférence a immédiatement brisé son ancien plafond de 1075m pour monter par paliers successifs (614m, 1411m) jusqu'à atteindre un pic d'exploitation de **1988m CPU**, soit l'utilisation quasi-intégrale (99,4 %) des 2000m alloués.
* **Maîtrise de l'empreinte mémoire** : L'initialisation du second worker a engendré une légère hausse de l'occupation RAM de base (passant de 441 Mi à 549 Mi), pour se stabiliser définitivement à **552 Mi**. Ce niveau de consommation laisse une marge de sécurité de 1,45 Gi avant d'atteindre le seuil critique d'OOMKill.
* **Stabilité des services connexes** : Le service de preprocessing s'est stabilisé à 56m CPU et 65 Mi de RAM au maximum de la charge, validant la pertinence de la redistribution des ressources au profit de l'inférence.

**Conclusion de la Séance 4 :** En maximisant l'usage du parallélisme au sein de la limite autorisée, le goulot d'étranglement lié au traitement séquentiel a été entièrement éliminé. Le système est désormais capable d'absorber la charge cible de 50 req/min de manière fluide, posant les bases de l'infrastructure pour le test de stress final à 150 req/min.

## Avant équilibrage

  RÉSULTATS  -  IMAGES / CHARGE

  Rate configuré    : 50 req/min
  Durée             : 300s
  Requêtes envoyées : 249
  Succès (HTTP 200) : 202
  Échecs            : 47
  Taux de succès    : 81.1%
  Latence moyenne   : 20.386s
  Latence P95       : 30.0s
  Latence max       : 30.0s


## Après équilibrage

  RÉSULTATS  -  IMAGES / CHARGE

  Rate configuré    : 50 req/min
  Durée             : 300s
  Requêtes envoyées : 250
  Succès (HTTP 200) : 250
  Échecs            : 0
  Taux de succès    : 100.0%
  Latence moyenne   : 7.588s
  Latence P95       : 21.875s
  Latence max       : 25.359s

* **Indicateurs de performance finaux (Palier 50 req/min)** : 
  * Requêtes traitées : 250 / 250
  * Taux de succès : 100.0 % (0 échec)
  * Latence moyenne : 7,588s (P95 : 21,875s)

# <span style="color: #d22c19;">Cinquième Séance : Évaluation des Limites Système sous Stress Élevé (150 req/min)</span>

## Analyse des Métriques en Régime de Saturation CPU
Pour identifier la capacité de rupture maximale de l'infrastructure mise en place, un stress test ultime a été mené à une cadence de 150 requêtes par minute sur une période de 300 secondes (soit une cible de 750 requêtes injectées).

Les relevés de consommation système (`kubectl top pods`) extraits à intervalles réguliers ont mis en lumière le comportement de saturation suivant :
* **Plafonnement CPU Absolu de l'Inférence** : Après une phase d'amorce, le service d'inférence est venu s'écraser de manière continue contre sa limite physique, oscillant strictement entre **1998m et 2003m CPU**. Cela valide le fonctionnement nominal et étanche de la directive `limits.cpu: 2000m` supervisée par Kubernetes.
* **Comportement de l'Infrastructure Connexe** : Le service de preprocessing a vu sa consommation se stabiliser entre 73m et 95m CPU pour 87 Mi de RAM, prouvant que la brique d'ingestion d'images disposait d'une réserve suffisante et n'était pas la cause du ralentissement. Le monitoring est resté linéaire à 2m CPU.
* **Stabilité de la Mémoire** : L'allocation RAM du conteneur d'inférence s'est maintenue entre 554 Mi et 557 Mi. L'absence de dérive ou d'explosion de la mémoire confirme la parfaite étanchéité du code face aux fuites de mémoire (*memory leaks*), écartant toute défaillance de type `OOMKilled`.

## Résultats du Palier de Stress et Interprétation
Les indicateurs de performance consolidés en fin de run affichent la rupture du pipeline :
* **Taux de succès** : 51,5 % (383 requêtes traitées avec succès, 361 échecs).
* **Profil de Latence** : Moyenne établie à 26,164s, avec un P95 et une latence maximale bloqués à **30,023s**.

L'interprétation de ces résultats met en évidence le phénomène de **Throttling CPU strict** théorisé dans le cours. N'ayant pas l'autorisation d'allouer plus de 2000m CPU au pod d'inférence (limite du `LimitRange`) ni de dépasser le `ResourceQuota` global du namespace, les calculs de tenseurs PyTorch ont été bridés par le noyau du cluster. Ce ralentissement induit a provoqué l'empilement des requêtes dans les buffers réseaux de l'API. N'ayant pu être traitées à temps, 48,5 % des requêtes ont expiré en atteignant le *timeout* client de 30 secondes.

## Conclusion Globale de l'Étude de Charge
Les expérimentations menées démontrent la robustesse et les limites de la configuration du Cas 1 :
1. L'infrastructure est **hautement résiliente** : aucun crash de conteneur, aucune corruption de mémoire ni perte de service n'a été déplorée, prouvant la stabilité de la conteneurisation.
2. Le point de bascule opérationnel se situe entre 50 req/min (100 % de succès, latence moyenne de 7,5s) et 150 req/min.
3. Pour franchir ce palier de stress en environnement de production réel, l'architecture exigerait soit une révision à la hausse du `ResourceQuota` (passage à 4 cœurs minimum pour permettre un passage à `--workers 4` applicatifs), soit la mise en œuvre d'un mécanisme de mise à l'échelle automatique des pods (*Horizontal Pod Autoscaler - HPA*).

### Tableau Comparatif des Performances et de la Consommation Système

| Métrique / Indicateur | Test 1 : Régime Nominal | Test 2 : Montée en Charge (Avant Optimisation) | Test 2 : Montée en Charge (Après Optimisation) | Test 3 : Stress Test Ultime |
| :--- | :---: | :---: | :---: | :---: |
| **Cadence configurée** | 10 req/min | 50 req/min | 50 req/min | 150 req/min |
| **Requêtes envoyées** | 50 | 250 | 250 | 744 |
| **Taux de succès (HTTP 200)** | **100.0 %** (50) | **80.4 %** (201) | **100.0 %** (250) | **51.5 %** (383) |
| **Taux d'échec (Timeout)** | **0.0 %** (0) | **19.6 %** (49) | **0.0 %** (0) | **48.5 %** (361) |
| **Latence Moyenne** | 0,684s | 15,278s | 7,588s | 26,164s |
| **Latence P95 / Max** | 1,387s / 5,485s | 30,000s / 30,000s | 21,875s / 25,359s | 30,000s / 30,023s |
| **CPU Inférence (`limits.cpu`)** | 107m (sur 1500m) | 1075m (Plafond mono-thread) | **1988m** (sur 2000m) | **2000m** (Saturation stricte) |
| **RAM Inférence (`limits.memory`)**| 449 Mi (sur 2 Gi) | 447 Mi (sur 2 Gi) | 552 Mi (sur 2 Gi) | 557 Mi (sur 2 Gi) |
| **CPU Preprocessing** | 11m (sur 1000m) | 55m (sur 1000m) | 56m (sur 1000m) | 92m (sur 1200m) |
| **RAM Preprocessing** | 82 Mi | 94 Mi | 65 Mi | 83 Mi |
| **Comportement du Système** | Stabilisation parfaite. Pic initial dû au *cold start* PyTorch. | **Goulot d'étranglement** applicatif (mono-thread Uvicorn). File d'attente saturée. | **Succès total**. Pleine exploitation du CPU alloué grâce au passage à `--workers 2`. | **Rupture par Throttling CPU**. Saturation physique de la limite autorisée par le YAML. |