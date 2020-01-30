####       UNIVERSIDAD INDUSTRIAL DE SANTANDER
#### ESCUELA DE ESTUDIOS INDUSTRIALES Y EMPRESARIALES
####          OPALO GRUPO DE INVESTIGACIÓN

#### PROPUESTA DE SOLUCIÓN DEL CVRP POR MEDIO DE Q-LEARNING

# Stiven Jaramillo & Elkin Rivera
# Enero del 2020

# NOTE Instancia A-n55-k9, tomada de http://vrp.galgos.inf.puc-rio.br/index.php/en/plotted-instances?data=A-n55-k9

""" Librerias """
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

""" Matrices """
# Archivo Excel con los nodos de la instancia en distancias euclidianas en la hoja "r" y demandas en la hoja "d"
file = "2.Matrices_instancia_media_54cl.xlsx"
r = pd.read_excel(file, sheet_name="r", header=None).values.astype(int)
q = np.zeros(r.shape, dtype=int)
demanda = pd.read_excel(file, sheet_name="d", header=None).values.astype(int)

""" Parámetros de la instancia """
# Siendo esta una instancia de 54 clientes, 9 vehículos de capacidad 100 y con un valor de solución óptimo de 1073
optimo = 1073
vehi = 9
cap_veh = 100
reward = 162  # Valor de recompensa
dimension = r.shape[0]
client = dimension - 2
best_costo = optimo * 100  # Costo grande para comparar el aprendizaje
demanda_total = np.sum(demanda)  # Totalidad de la demanda a saldar

""" Parámetros del problema """
np.random.seed(6865)  # NOTE Valor de la semilla
epochs = 192  # Corridas

""" Factores """

alpha = 0.7  # Tasa de aprendizaje (D.E. 0.7)
gamma = 0.01  # Factor de descuento (D.E. 0.01)
lambda_e = 0.3  # Factor de descomposición (D.E. 0.3)

""" Extras """
g_print = True  # True si se desea graficar
vector_export = False  # True si se desea exportar a csv best_scores
scores_export = False  # True si se desea exportar a csv scores

epsilon = 1  # 100% exploracion, 0% explotación
# Cambio de fuerte exploración a fuerte explotación
epsilon_decay = 1 - (lambda_e)
epsilon_min = 0.1  # Después de un largo tiempo 10% exploracion, 90% explotacion

""" Entrenamiento """
best_score = np.array([], dtype=int)  # Vector de costo minimo
best_ruta = np.array([], dtype=int)  # Vector de la mejor ruta
scores = np.array([], dtype=int)  # vector de costo por corrida

""" Funciones """
# Acción a tomar según la política ε-greedy, dilema entre exploración y Explotación
def egreedy(epsilon):
    if (
        np.random.rand() <= epsilon
    ):  # Exploracion, tomando un cliente completamente al azar
        acc = np.random.choice(np.where(r[estado] < 0)[0])
    else:  # Explotacion, toma el cliente de mayor valor Q
        max_q = np.max(np.take(q[estado], np.where(r[estado] < 0)[0]))
        acc = np.random.choice(
            np.intersect1d(
                np.where((q[estado] == max_q))[0], np.where(r[estado] < 0)[0]
            )
        )
    return acc


# Mayor valor Q del estado resultado de haber seleccionado la acción "acc"
# sin tener en cuenta del depósito por aún tener demanda para satisfacer clientes
def max_tn(acc):
    max_q_tn = np.max(np.take(q[acc], np.where(r[acc] != 0)[0]))
    return max_q_tn


# Generalized Q-Learning (Atienza p.287)
def update(estado, acc, alpha, gamma, max_q_tn):
    upd = q[estado, acc] + alpha * (r[estado, acc] + gamma * max_q_tn - q[estado, acc])
    return upd


for i in range(epochs):

    estado = 0  # Comienzo del vehículo en el depósito
    cap = cap_veh  # Capacidad vehicular
    d = demanda.copy()  # Vector demanda de clientes

    while cap >= min(d[d > 0]):  # | (len(d[d > 0]) == 0): # NOTE revisar

        acc = egreedy(epsilon)  # Desición de cuál cliente visitar
        max_q_tn = max_tn(acc)  # Mayor valor Q partiendo de la acción tomada
        q[estado, acc] = update(estado, acc, alpha, gamma, max_q_tn)  # Actualización
        if cap >= d[acc]:  # Saldar a un cliente es tener suficiente capacidad
            cap = cap - d[acc]  # Reudcción de la capacidad del vehículo
            d[acc] = 0  # Cliente saldado
        estado = acc  # En este caso el nuevo estado es la acción tomada

    acc = dimension - 1
    max_q_tn = 0
    q[estado, acc] = update(estado, acc, alpha, gamma, max_q_tn)  # Actualización

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    """ Prueba de ruteo """
    ruta = np.array([], dtype=int)
    costo_q = 0
    for j in range(vehi):
        estado_q = 0
        ruta = np.append(ruta, estado_q)
        capac_q = cap_veh
        v_clientes = np.ones(dimension, dtype=int)
        v_clientes[[0, -1]] = 0
        v_clientes[ruta] = 0
        dema = demanda.copy()
        # min_dema = np.min(dema[dema > 0])
        l_clientes = np.where(v_clientes == 1)[0]

        while (
            len(np.intersect1d(np.where(capac_q >= dema)[0], l_clientes)) > 0
        ):  # Mientras haya clientes disponibles

            l_max_q = np.where(
                q[estado_q]
                == np.max(
                    np.take(
                        q[estado_q],
                        np.intersect1d(np.where(capac_q >= dema)[0], l_clientes),
                    )
                )
            )[
                0
            ]  # valores Q iguales al max Q de clientes disponibles
            ax = np.random.choice(
                np.intersect1d(
                    l_max_q, np.intersect1d(np.where(capac_q >= dema)[0], l_clientes)
                )
            )  # Escoge un cliente de los que tiene max Q de los disponibles
            ruta = np.append(ruta, ax)
            capac_q = capac_q - dema[ax][0]
            costo_q = costo_q - r[estado_q, ax]
            v_clientes[ax] = 0
            l_clientes = np.where(v_clientes == 1)[0]
            estado_q = ax
        ax = dimension - 1
        costo_q = costo_q - r[estado_q, ax] + reward

    min_costo = costo_q.copy()
    if min_costo < best_costo:
        best_costo = min_costo
        best_ruta = ruta
        pr = i + 1

    scores = np.append(scores, costo_q / optimo)
    best_score = np.append(best_score, best_costo / optimo)

""" Mostrar resultados (Rutas, demandas, costo ruteo, epoch del minimo) """
d = np.split(best_ruta, np.where(best_ruta == 0)[0])
z = 0
Demanda_satis = 0
for l in d:
    if len(l) != 0:
        z += 1
        dema_sas = np.sum(np.take(demanda, l[1:]))
        Demanda_satis = Demanda_satis + dema_sas
        print("Ruta #", z, l[1:], ", satisface:", dema_sas)
print("")
print(
    "Demanda satisfecha:",
    Demanda_satis,
    "(",
    "{0:.0%}".format(Demanda_satis / demanda_total),
    "de la demanda total )",
)
print("")
print(
    "Costo minimo obtenido:",
    best_costo,
    "( %.3f" % (best_costo / optimo),
    "veces el optimo )",
    end="\n\n",
)
print("Encontrado en epoch #", pr)

""" Exportar CSV """
if vector_export is True:
    vector_path = (
        # "Instancia media (54 clientes)\\mediana_alpha_"
        "8. Código definitivo\\Instancia media (54 clientes)\\Vectors\\mediana_alpha_"
        + str(alpha)
        + "_gamma_"
        + str(gamma)
        + "_lambda_"
        + str(lambda_e)
        + "_epochs_"
        + str(epochs)
        + ".csv"
    )
    np.savetxt(vector_path, best_score, delimiter="|")

""" Gráficas """
if g_print is True:
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(16, 8))
    plt.plot(scores, label="Costo", color="red", alpha=0.2, lw=2)
    plt.plot(
        best_score, label="Costo mínimo obtenido", color="darkred", alpha=0.7, lw=2
    )
    plt.xlabel("Corridas")
    plt.ylabel("Costo de ruta / Costo de ruta óptima")
    plt.title("Q-Learning para solución al CVRP con instancia de 54 clientes")
    plt.legend()
    plt.show()

""" Exportar score """
if scores_export is True:
    score_path = (
        # "Instancia media (54 clientes)\\med_score_alpha_"
        "8. Código definitivo\\Instancia media (54 clientes)\\Vectors\\med_score_alpha_"
        + str(alpha)
        + "_gamma_"
        + str(gamma)
        + "_lambda_"
        + str(lambda_e)
        + "_epochs_"
        + str(epochs)
        + ".csv"
    )
    np.savetxt(score_path, scores, delimiter="|")
