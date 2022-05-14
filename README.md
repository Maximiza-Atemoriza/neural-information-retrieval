# Neural-Information-Retrieval

- Adrián Rodríguez Portales C512
- David Guaty Domínguez C512
- Rodrigo Daniel Pino Trueba C512

## Instalación 

```bash
pip install requirements.txt
```
### CLI

```bash
python main.py <DATASET> <RANKING>
```
`<DATASET`: dataset sobre el cual se van a ejecutar las consultas ( por ahora solo está disponible CRAN )

`<RANKING>`: cantidad de documentos recuperados

Para más información: 

```bash
python3 main.py --help
```

**nota**:En la fase actual del proyecto se usa el modelo vectorial para la representación y recuparación de los documentos.

### GUI

**TODO**

### REST API

**TODO**

## Estructura e implementación

El proyecto está formado por dos módulos fundamentales: `parsers` y `information_retrieval`. En el primero se encuentran todas las funciones relacionadas con el procesamiento de los dataset de ejemplo ( en la actual implementación solo es posible trabajar con CRAN ). En el segundo se encuentran los principales componentes del sistema de recuperación de información. En la fase actual del proyecto solo se tiene implementado el modelo vectorial visto en clase, el cual servirá de base para modelo basado en redes neurales que se va a desarrollar. 
