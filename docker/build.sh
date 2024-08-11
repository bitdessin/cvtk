#!/bash/bin

sudo env USERNAME="$(id -un)" docker compose build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg USERNAME="$(id -un)" \
    --build-arg GROUPNAME="$(id -un)"
