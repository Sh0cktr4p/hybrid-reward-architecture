user=${1:-user}
echo "Chosen mode: $user"
if [ "$user" = "root" ]
then
    docker run -it \
        --net=host \
        --volume="$(pwd)/:/root/hybrid-reward-architecture/" \
        hybrid-reward-architecture-train/root:v2
elif [ "$user" = "user" ]
then
    docker run -it \
        --net=host \
        --volume="$(pwd)/:/home/$USER/hybrid-reward-architecture/" \
        hybrid-reward-architecture-train/$USER:v2
else
    echo "User mode unknown. Please choose user, root, or leave out for default user"
fi
