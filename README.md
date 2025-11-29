# Proyecto_Taller_GPU

Ejeplo de uso:

Para aplicar un filtro al video por defecto (test.mp4)

```bash
make {filtro}
```
Por ejemplo para aplicar filtro ascii
```bash
make ascii # Sin audio

make ascii_audio # Con audio
```

Si se quiere utilizar un video personalizado:

```bash
make {filtro} {nombre archivo .mp4}
```
Ejemplo de filtro grayscale a **video_custom.mp4**
```bash
make grayscale INPUT=video_custom.mp4 # Sin audio

make grayscale_audio INPUT=video_custom.mp4 # Con audio
```