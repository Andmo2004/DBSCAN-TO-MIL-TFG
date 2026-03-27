# NOTAS Y APUNTES A TENER EN CUENTA EN EL PROYECTO
## CONCLUSIONES DURANTE LAS PRUEBAS
### Distancias Favoritas por Dataset
| HAUSDORFF | CAUCHY SCHWARZ |
|:-:|:-:|
| musk1 | BirdsChestnut-backedChickadee |
| musk2 | BirdshammondsFlycatcher |
| mutagenesis3_atoms |   |
| mutagenesis3_chains |   |
| Newsgroups1 |   |
| simple_dummy |   |
| Thioredoxin |   |
| ImageElephant |   |
| Harddrive1 |   |    

BirdsChestnut/BirdsHammonds — estos datasets de audio son intrínsecamente difíciles para Hausdorff porque las bolsas tienen distribuciones de instancias muy solapadas. Para esos, la métrica cauchy_schwarz puede funcionar mejor.

Patrón 1: F1=0, Specificity=1 → Todo predicho como Negativo
BirdsChestnut, BirdsHammonds, Thioredoxin
El majority voting mapea todos los clusters a clase 0. Ocurre cuando la clase negativa domina tanto que incluso clusters "mixtos" se asignan a 0.


Patrón 2: Recall=1, Specificity=0 → Todo predicho como Positivo
ImageElephant
El caso inverso: todos los clusters mapean a clase 1.
Patrón 3: Clusters=1 → Sin separación real
BirdsChestnut (1 cluster), BirdsHammonds (1 cluster), Newsgroups1 (1 cluster)
Epsilon demasiado grande: todo queda en un blob.

Los dos problemáticos no mejoraron porque el fix de imbalance_ratio no llegó a activarse — 
ambos tienen min_pts=10 y min_pts=8 respectivamente, lo que indica que el grid search sí exploró distintas 
configuraciones pero todas terminaron con el mismo resultado. El problema es más profundo.

| Dataset                  | F1 iter1 | F1 iter2 | F1 iter3 | F1 actual | Tendencia                         |
|--------------------------|----------|----------|----------|-----------|-----------------------------------|
| BirdsHammonds            | —        | 0.000    | 0.667    | 0.939     | ✅ Mejorando                      |
| Harddrive1               | —        | 0.708    | 0.708    | 0.943     | ✅ Mejorando                      |
| mutagenesis3_atoms       | —        | 0.829    | 0.829    | 0.894     | ✅ Mejorando                      |
| mutagenesis3_chains      | —        | 0.831    | 0.831    | 0.746     | 🔴 Empeoró                        |
| Newsgroups1              | —        | 0.471    | 0.786    | 0.786     | 🟡 Estable                        |
| musk1                    | 0.722    | 0.722    | 0.720    | 0.720     | 🟡 Estable                        |
| musk2                    | 0.640    | 0.640    | 0.640    | 0.640     | 🟡 Estancado                      |
| ImageElephant            | —        | 0.667    | 0.667    | 0.667     | 🟡 Estable                        |
| BirdsChestnut            | —        | 0.000    | 0.051    | 0.208     | ✅ Mejorando (pero techo bajo)    |
| Thioredoxin              | —        | 0.000    | 0.167    | 0.154     | 🟡 Sin mejora real                |

Hay un retroceso real: mutagenesis3_chains bajó de 0.831 a 0.746. El grid search eligió min_pts=3 con 14 clusters en lugar de min_pts=2 con 7 clusters que daba mejor resultado. El score interno no está capturando bien este caso.

El problema es la penalización de fragmentación: con frag_penalty = max(0, n_clusters - 10) * 0.01, 14 clusters solo penaliza 0.04, insuficiente para compensar la diferencia de F1.

Para musk2 estancado en 0.640 con 10 clusters: el problema es que 9-10 clusters con Hausdorff para 71 bolsas indica que la métrica no crea separación natural entre positivos y negativos en ese espacio. Podría valer la pena probar cauchy_schwarz para musk2.

quedan por mejorar (BirdsChestnut, Thioredoxin, musk2)

| Dataset               | Std+Hau | Std+CS | MM+Hau | MM+CS | Mejor                 |
|-----------------------|--------:|-------:|-------:|------:|-----------------------|
| musk1                 | 0.720   | —      | 0.769  | 0.522 | MM+Hau (0.769)        |
| musk2                 | 0.640   | —      | 0.667  | 0.667 | MM+CS o MM+Hau (0.667)|
| ImageElephant         | 0.667   | —      | 0.675  | 0.753 | MM+CS (0.753)         |
| BirdsChestnut         | 0.208   | —      | 0.100  | 0.146 | Std+CS (0.208)        |
| BirdsHammonds         | 0.939   | —      | 0.984  | 0.000 | MM+CS (0.984)         |
| Harddrive1            | 0.943   | —      | 0.941  | 0.984 | MM+CS (0.984)         |
| mutagenesis_atoms     | 0.894   | —      | 0.865  | 0.857 | Std+Hau (0.894)       |
| mutagenesis_chains    | 0.746   | —      | 0.702  | 0.816 | MM+CS (0.816)         |
| Newsgroups1           | 0.786   | —      | 0.667  | 0.593 | Std+Hau (0.786)       |
| simple_dummy          | 0.889   | —      | 0.889  | 0.889 | Empate                |
| Thioredoxin           | 0.154   | —      | 0.133  | 0.333 | MM+CS (0.333)         |

MinMaxScaler + Cauchy-Schwarz es la combinación ganadora en 5 de 11 datasets. Tiene sentido porque Cauchy-Schwarz mide similitud de orientación entre vectores, y MinMax garantiza que todas las features contribuyan proporcionalmente al producto interno — StandardScaler puede crear features con media cero que distorsionan la similitud coseno.

StandardScaler + Hausdorff sigue siendo mejor para datasets con features gaussianas y bolsas donde importa la distancia absoluta entre instancias extremas (mutagenesis, Newsgroups1), porque Hausdorff es sensible a outliers y StandardScaler los controla mejor que MinMax.

Thioredoxin sube de 0.154 a 0.333 con MM+CS — sigue siendo el dataset más difícil pero hay mejora real. Con Precision=1.0 y Recall=0.20, hay 8 clusters de los cuales probablemente solo 1-2 contienen positivos reales y están bien identificados, pero la mayoría de positivos siguen dispersos en clusters negativos o en ruido (40.74%).



|SED — musk1|
|-----------------------|
|  Clusters : 8 |
|  Ruido    : 40.6%|
|  SED      : 33.854300  (↓ mejor)|

Tener 40.6% de ruido es bastante alto. Significa que casi la mitad de las bolsas no quedaron asignadas a ningún cluster, y esas bolsas no contribuyen al SED. Esto es importante tenerlo en cuenta para comparar SED entre configuraciones, porque un modelo con menos ruido y SED ligeramente mayor puede estar describiendo mejor el dataset que uno con SED bajo pero mucho ruido.