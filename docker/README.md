# Docker scripts
Here are some scripts that can make using this project easier.\

Install Docker:

```bash 
# It's preferably to use -n flag that enables Nvidia drivers support.
bash install_docker.sh -n # (Re)install Docker
bash build_docker.sh -n # Build Docker container:
bash run_docker.sh -n # Run Docker container
```

You can access the running container:
```bash
bash into_docker.sh
```

## WSL + NVIDIA OpenGL

Для запуска контейнера в WSL2 с использованием NVIDIA драйвера:

1. Соберите образ с поддержкой GPU (при необходимости):
	```bash
	bash build_docker.sh -n
	```
2. Запустите контейнер в режиме Wayland + NVIDIA, который автоматически пробрасывает библиотеки `/usr/lib/wsl/lib`, устройство `/dev/dxg` и сокеты WSLg:
	```bash
	bash run_docker.sh -wn
	```
3. Внутри контейнера можно убедиться, что OpenGL использует NVIDIA, выполнив, например:
	```bash
	glxinfo | grep "OpenGL renderer"
	```

Скрипт `run_docker.sh` сам проверяет наличие необходимых библиотек и, в случае WSL, настраивает переменные окружения (`__GLX_VENDOR_LIBRARY_NAME`, `__NV_PRIME_RENDER_OFFLOAD`, `__VK_LAYER_NV_optimus`) и `LD_LIBRARY_PATH`, чтобы OpenGL работал через драйвер NVIDIA.
