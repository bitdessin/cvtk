#!/bash/bin

sudo USERNAME="$(id -un)" docker compose run -it --rm \
    -e UID="$(id -u)" \
    -e GID="$(id -g)" \
    -e USERNAME="$(id -un)" \
    -e GROUPNAME="$(id -un)" \
    devel-env
