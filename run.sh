docker run  \
    --gpus all \
    -v `pwd`/code:/home/ubuntu  \
    --rm \
    -p 11311:11311 -p 9090:9090 -p 5900:5900 -p 6080:6080 \
    veggiebenz/rbx1_bullet:v3 &

# -u $(id -u):$(id -g)

dockerpid=$!
echo "docker container pid is $dockerpid"

sleep 7

#xdg-open http://localhost:6080/vnc.html &
xdg-open http://localhost:6080/vnc_auto.html &
#xdg-open http://localhost:9090 &

# pop a new terminal tab connected to the docker instance
gnome-terminal --tab --title="DOCKER" -- bash -c 'docker exec -it `docker ps --format "{{.Names}}"` /bin/bash --rcfile /home/ubuntu/.bashrc  '

trap "kill $dockerpid" INT

wait
