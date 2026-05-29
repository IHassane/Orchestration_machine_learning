#!/bin/bash
echo "🚀 Fixation des ports de la stack MLOps..."

# Suppression des anciens tunnels s'ils existent
pkill -f "port-forward" 2>/dev/null

# Lancement des deux port-forwards en tâche de fond
kubectl port-forward svc/preprocessing-svc 8001:8001 -n projet-trigramme > /tmp/k8s_preprocess.log 2>&1 &
kubectl port-forward svc/monitoring-svc 9090:9090 -n projet-trigramme > /tmp/k8s_monitoring.log 2>&1 &

# Petite pause pour laisser à K8s le temps d'initier la connexion
sleep 2

echo "🔍 Vérification des processus :"
ps aux | grep "port-forward svc/" | grep -v grep

echo "------------------------------------------------"
echo "✅ Si deux lignes s'affichent au-dessus, c'est tout bon !"
