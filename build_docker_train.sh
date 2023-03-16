user=${1:-user}
echo "Chosen mode: $user"

if [ "$user" = "root" ]
then
    DOCKER_BUILDKIT=1 docker build \
        -f Dockerfile.train \
        --build-arg MODE=root \
        -t hybrid-reward-architecture-train/root:v2 .
elif [ "$user" = "user" ]
then
    DOCKER_BUILDKIT=1 docker build \
        -f Dockerfile.train \
        --build-arg MODE=user \
        --build-arg USER_UID=$(id -u) \
        --build-arg USER_GID=$(id -g) \
        --build-arg USERNAME=$USER \
        -t hybrid-reward-architecture-train/$USER:v2 .
else
  echo "User mode unkown. Please choose user, root, or leave it out for default user"
fi
