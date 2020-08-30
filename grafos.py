import hashlib
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from igraph import *
import stanza
import seaborn as sns
import itertools
import randomcolor
import calendar
import matplotlib.dates as mdates
import locale


def createFile(communities, filename):
    f = open("files/" + filename + ".txt", "w+")
    i = 1
    for community in communities:
        f.write("Comunidad " + str(i) + ":\n\n")
        i+=1;
        for hashtag in community:
            f.write(hashtag + "\n")
        f.write("\n\n")
    f.close()


# Obtiene el nodo de mayor grado del subgrafo, y le asigna la etiqueta en el nodo correspondiente del grafo principal
def setMaxDegreeLabel(subgraph, graph):
    topDegree = max(subgraph.degree())
    topDegreeNode = subgraph.vs.find(_degree=topDegree)
    topDegreeGeneralGraph = graph.vs.find(name=topDegreeNode["name"])
    topDegreeGeneralGraph["label"] = topDegreeGeneralGraph["name"]

# Crea un grafo a partir de una lista de aristas y lo devuelve por parámetro
def creategraph(edges):
    # Se crea una lista de vertíces y se ordena
    vertices = set()
    for line in edges:
        vertices.update(line)
    vertices = sorted(vertices)

    # Se crea un grafo vacío
    g = Graph()

    # Se añaden los vertíces al grafo
    g.add_vertices(vertices)

    # Se añaden las aristas al grafo
    g.add_edges(edges)

    # Se fijan todos los pesos a 1
    g.es["weight"] = 1

    # Se unifican las aristas múltiples y se suman sus pesos
    g.simplify(combine_edges={'weight': 'sum'})


    return g

# Genera un fichero png con el grafo pasado por parámetro
def plotgraph(graph, filename, showlabel):

    visual_style = {}
    # Se calcula el grado de cada vértice para asignarle un tamaño proporcional a éste
    degrees = []
    vertexSize = []
    for row in graph.degree():
        degrees.append(row)
        vertexSize.append(math.sqrt(row) * 7)

    if showlabel == True:
        degrees.sort(reverse=True)
        topdegrees = list(set(degrees[:10]))
        topdegrees.sort(reverse=True)

        for i in topdegrees:
            # Para aquellos nodos con grado i se asigna como etiqueta su nombre para mostrar el hashtag
            nodos = graph.vs.select(_degree=i)
            for nodo in nodos:
                nodo["label"] = nodo["name"]

    # Se fija el tamaño del vértice
    visual_style["vertex_size"] = vertexSize

    # Se asigna nombre al fichero
    out_fig_name = 'figures/' + filename + '.png'

    # Set bbox and margin
    visual_style["bbox"] = (3000, 3000)
    visual_style["margin"] = 17

    # Set edge witdh
    widths = []
    if showlabel==False:
        for weight in graph.es["weight"]:
            if (weight < 10):
                widths.append(weight)
            else:
                widths.append(10)
        visual_style["edge_width"] = widths

    # Set vertex label attributes
    visual_style["vertex_label_size"] = 30
    visual_style["vertex_label_font"] = 4

    # Don't curve the edges
    visual_style["edge_curved"] = False

    # Pintar cada nodo de un color en funcion de su grado
    node_colours = []
    for i in graph.degree():
        if i == 1:
            node_colours.append("grey")
        elif i == 2:
            node_colours.append("yellow")
        elif i == 3:
            node_colours.append("red")
        elif i == 4:
            node_colours.append("blue")
        elif i == 5:
            node_colours.append("orange")
        elif i == 6:
            node_colours.append("pink")
        elif i == 7:
            node_colours.append("purple")
        elif i == 8:
            node_colours.append("light green")
        elif i == 9:
            node_colours.append("tan")
        elif i == 10:
            node_colours.append("magenta")
        else:
            node_colours.append("sky blue")

    visual_style["vertex_color"] = node_colours

    # Set the layout
    my_layout = graph.layout_fruchterman_reingold()
    # my_layout = graph.layout_fruchterman_reingold() #Esta distribución permite visualizar el grafo completo
    visual_style["layout"] = my_layout

    # Plot the graph
    plot(graph, out_fig_name, **visual_style)

    # Genera un fichero png con el grafo pasado por parámetro

def plotcommunities(graph, filename, clusters):
    communities = []
    visual_style = {}
    # Calculamos el grado de cada vértice para asignarle un tamaño proporcional a éste
    degrees = []
    vertexSize = []
    for row in graph.degree():
        degrees.append(row)
        vertexSize.append(math.sqrt(row) * 7)

    degrees.sort(reverse=True)
    topdegrees = list(set(degrees[:5]))
    topdegrees.sort(reverse=True)

    for i in topdegrees:
        # Para aquellos nodos con grado i se asigna como etiqueta su nombre para mostrar el hashtag
        nodos = graph.vs.select(_degree=i)
        for nodo in nodos:
            nodo["label"] = nodo["name"]

    # Set vertex size
    visual_style["vertex_size"] = vertexSize

    # Visualise the Graph
    out_fig_name = 'figures/' + filename + '.png'

    # Set bbox and margin
    visual_style["bbox"] = (3000, 3000)
    visual_style["margin"] = 17

    # Set vertex label attributes
    visual_style["vertex_label_size"] = 30
    visual_style["vertex_label_font"] = 4

    # Don't curve the edges
    visual_style["edge_curved"] = False

    # Pintar cada nodo de un color en funcion de su grado
    node_colours = []
    for i in range(len(clusters)):
         rand_color = randomcolor.RandomColor()
         node_colours.extend(rand_color.generate())

    lengths = []
    for cluster in clusters:
        lengths.append(len(cluster))

    """ ANALIZAR ESTA PARTE EN EL CASO DE QUE DESPUES SE APLIQUE CLUSTERS SOLO A GIANT
    subgraphs = clusters.subgraphs()

    for clid, subgraph in enumerate(subgraphs):
        if subgraph.vcount() != max(lengths):
            setMaxDegreeLabel(subgraph, graph)
            #Asignar mismo color a la comunidad
            names = []
            for node in subgraph.vs:
                name = node["name"]
                names.append(name)
                graph.vs.select(name=name)["color"] = node_colours[clid]
            communities.append(names) """

    # Se obtiene la componente conexa
    #giant = graph.clusters().giant()

    # ALGORITMO 1: edge betweenness
    dendrogram = graph.community_edge_betweenness()
    clustersGiant = dendrogram.as_clustering()

    # ALGORITMO 2: walktrap
    #wtrap = graph.community_walktrap(weights=graph.es["weight"], steps=4)
    #wtrap = graph.community_walktrap(steps=4)
    #clustersGiant = wtrap.as_clustering()

    # ALGORITMO 3: label propagation
    #clustersGiant = graph.community_label_propagation(weights=graph.es["weight"])
    #clustersGiant = graph.community_label_propagation()

    node_colours = []
    for i in range(len(clustersGiant)):
        rand_color = randomcolor.RandomColor()
        node_colours.extend(rand_color.generate())

    subgraphs = clustersGiant.subgraphs()
    for clid, subgraph in enumerate(subgraphs):
        names = []
        for node in subgraph.vs:
            setMaxDegreeLabel(subgraph, graph)
            # Asignar mismo color a comunidad
            name = node["name"]
            names.append(name)
            graph.vs.select(name=name)["color"] = node_colours[clid]
        communities.append(names)

    communities.sort(key=len,reverse=True)
    communities2 = []
    for community in communities:
        if len(community) > 3:
            communities2.append(community)
    createFile(communities2, "grafoHashtags")

    # Set the layout
    my_layout = graph.layout_fruchterman_reingold()
    visual_style["layout"] = my_layout

    # Plot the graph
    # plot(clustersGiant, out_fig_name, mark_groups = True,  **visual_style)

    return communities2

# Genera un fichero png con un gráfico de barras
def plotbarchart(numberbars, x, y, title, xlabel, ylabel, filename):
    sns.set()
    plt.figure(figsize=(9, 6))
    plt.bar(x=x[:numberbars], height=y[:numberbars], color='midnightblue')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=45)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/' + filename + '.png')


# Extraer palabras más usadas a partir de un documento
def extractwords(doc):
    words = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ('NOUN', 'ADJ', 'VERB') and word.text.startswith("\\") == False:
                words.append(word.text)

    wordslist = [x.lower() for x in words]

    #Se descartan las palabras relacionadas con ciencia ciudadana
    wordslist = [word for word in wordslist if word not in stopwords]

    wordslist = np.unique(wordslist, return_counts=True)

    return wordslist

def getDays(df):
    df = pd.to_datetime(df, format="%d/%m/%Y %H:%M").dt.date
    days = pd.unique(df)
    days.sort()
    return days

def getMonths(df):
   # df = pd.to_datetime(df, format="%d/%m/%Y %H:%M").dt.month
    df = pd.to_datetime(df, format="%d/%m/%Y %H:%M").dt.date
    days = pd.unique(df)
    days.sort()
    months = []
    for day in days:
        month = day.month
     #   month = day.strftime("%M")
        months.append(month)
    months = list(set(months))


    locale.setlocale(locale.LC_TIME, 'esp')
    for idx, month in enumerate(months):
        months[idx] = calendar.month_name[month]

    print(months)
    return months

def plottemporalserie(days, df, elements, filename, title):
    df["Fecha"] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y %H:%M").dt.date
    numHashtag = []
    for hashtag in elements[:5]:
        numPerDay = []
        for day in days:
            dfOneDay = df[df['Fecha'] == day]
            count = dfOneDay['Texto'].str.contains(hashtag, case=False).sum()
            numPerDay.append(count)
        numHashtag.append(numPerDay)


    sns.reset_orig()
    fig = plt.figure(figsize=(9, 6))

    colours = ["red", "blue", "green", "orange", "magenta"]

    i = 0
    for hashtag in elements[:5]:
        plt.plot_date(days, numHashtag[i], colours[i], label=hashtag)
        i += 1

    # Set title and labels for axes
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel("Fecha", fontsize=15)
    plt.ylabel("Número de veces", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    fig.autofmt_xdate()
    plt.savefig("figures/" + filename + ".png", bbox_inches="tight")


start_time = time.time()

#Se lee el csv, separado por ; y omitiendo errores por si existieran filas erróneas
df = pd.read_csv('data/dataSetFinal.csv', sep=';', error_bad_lines=False)
stopwords = ['#citizenscience', 'citizenscience', 'rt', 'citizen', 'science', 'citsci','cienciaciudadana']

#Se eliminan los valores a null si existieran
df.dropna(inplace = True)


# CALCULAR GRAFO RTs
dfRT = df[['Usuario', 'Texto', 'Fecha']].copy() # Se copia a un dataframe de trabajo
dfRT = dfRT[dfRT['Texto'].str.match('RT: @')]  # Se seleccionan sólo las filas con RT

subset = dfRT[['Usuario', 'Texto']] # Se descarta la fecha
retweetEdges = [list(x) for x in subset.to_numpy()] # Se transforma en una lista

for row in retweetEdges:
    matchRT = re.search('@(\w+)', row[1]).group(1)  # Se extrae la primera mención que hace referencia a la cuenta retuiteada
    row[1] = hashlib.md5(matchRT.encode()).hexdigest()  # Convierte el nombre de la cuenta en hash y lo asigna al elemento

#grafoRT = creategraph(retweetEdges[:960]) #Se crea el grafo con el máximo número de elementos para que se pinte bien
grafoRT = creategraph(retweetEdges)
plotgraph(grafoRT, 'grafoRT', False)

# CALCULAR HASTAGHS EN RT PARA GRAFICA
dfHashtagsRT = df[['Usuario', 'Texto', 'Fecha']].copy()
dfHashtagsRT = dfHashtagsRT[dfHashtagsRT['Texto'].str.match('RT:')]
listHashtagsRT = dfHashtagsRT['Texto'].to_numpy()
print("Longitud RT: " + str(len(listHashtagsRT)))

hashtagsRT = []
for row in listHashtagsRT:
    match = re.findall('#(\w+)', row)
    for hashtag in match:
        hashtagsRT.append(hashtag)


hashtagsRT = [x.lower() for x in hashtagsRT]
hashtagsRT = [word for word in hashtagsRT if word not in stopwords]
hashtagsRT = np.unique(hashtagsRT, return_counts=True)

# Se ordena la lista de hashtags en función de mayor aparición y se recompone en 2 listas diferentes
hashtagsRT = sorted((zip(hashtagsRT[1], hashtagsRT[0])), reverse=True)
sortedNumberHashtags, sortedHashtagsRT = zip(*hashtagsRT)
print("Hashtags RT: " + str(len(sortedNumberHashtags)))

plotbarchart(10, sortedHashtagsRT, sortedNumberHashtags, 'Top 10 hashtag con más retweets',
             'Hashtag', 'Nº de veces', 'graficoHashtagsRT')

# CALCULAR GRAFO CITAS
dfMentions = df[['Usuario', 'Texto']].copy()

# Se descartan los RTs
dfEliminarRTs = dfMentions[dfMentions['Texto'].str.match('RT:')]
dfMentions.drop(dfEliminarRTs.index, axis=0, inplace=True)

# Se buscan las citas
dfMentions = dfMentions[dfMentions['Texto'].str.contains('@.')]
mentionsSubset = dfMentions[['Usuario', 'Texto']]

mentionsList = [list(x) for x in mentionsSubset.to_numpy()]
mentionEdges = []
for row in mentionsList:
    match = re.search('@(\w+)', row[1])  # Se extrae la primera mención
    if match:
        match = match.group(1)
        row[1] = hashlib.md5(match.encode()).hexdigest()  # Convierte el nombre en hash y lo asigna al elemento
        mentionEdges.append(row)

#mentionsGraph = creategraph(mentionEdges[:650])
mentionsGraph = creategraph(mentionEdges)
plotgraph(mentionsGraph, 'grafoCitas', False)

# GRAFO COMBINADO
combinedEdges = retweetEdges + mentionEdges
#combinedGraph = creategraph(combinedEdges[:950])
combinedGraph = creategraph(combinedEdges)
plotgraph(combinedGraph, 'grafoCombinado', False)


# CALCULAR HASTAGS PRINCIPALES SIN CONTAR RETWEETS Y CREAR GRAFO HASHTAGS
dfMainHashtags = df[['Usuario', 'Texto', 'Fecha']].copy()

dfEliminarHashtagsRT = dfMainHashtags[dfMainHashtags['Texto'].str.match('RT:')]
dfMainHashtags.drop(dfEliminarHashtagsRT.index, axis=0, inplace=True)

subsetMainHashtags = dfMainHashtags['Texto']
listMainHashtags = subsetMainHashtags.to_numpy()
print("Longitud Main: " + str(len(listMainHashtags)))

mainHashtags = []
aristasHashtags = []
# Extrae la lista de hashtags por cada tweet y relaciona todos los hashtags que aparecen en el mismo
for row in listMainHashtags:
    match = re.findall('#(\w+)', row.lower())
    length = len(match)
    try:
        match = [word for word in match if word not in stopwords]
     #   match.remove('citizenscience') # Si existe, se elimina de la lista el hashtag citizenscience
    except ValueError:
        pass

    # Se añaden una arista de cada hashtag con los siguientes hashtags que aparezcan
    for index, hashtag in enumerate(match):
            mainHashtags.append(hashtag)
            if index < (length-2):
                nextHashtags = match[index+1:length-1]
                for nextHashtag in nextHashtags:
                    aristasHashtags.append([hashtag, nextHashtag])

mainHashtags = np.unique(mainHashtags, return_counts=True)

# Se ordena la lista de hashtags en función de mayor aparición y se recompone en 2 listas diferentes
mainHashtags = sorted((zip(mainHashtags[1], mainHashtags[0])), reverse=True)
sortedNumberHashtags, sortedMainHashtags = zip(*mainHashtags)
print("Hashtags principales: " + str(len(sortedNumberHashtags)))

# gráfica hashtags más usados
#plotbarchart(10, sortedMainHashtags, sortedNumberHashtags, 'Top 10 hashtag principales más utilizados',
#           'Hashtag', 'Nº de veces', 'graficoHashtagsUsados')

# Se descartan los hashtags que sólo aparecen una vez
hashtagsOnce = [t[1] for t in mainHashtags if t[0] == 1]

# Se eliminan las aristas que tengan como alguno de sus nodos los hashtags con una sola aparición
hashtagsFinales = [hashtag for hashtag in aristasHashtags if hashtag[0] not in hashtagsOnce]
hashtagsFinales = [hashtag for hashtag in hashtagsFinales if hashtag[1] not in hashtagsOnce]

# Grafo hashtags relacionados
#hashtagsGraph = creategraph(hashtagsFinales[:22000])
hashtagsGraph = creategraph(hashtagsFinales)
#plotgraph(hashtagsGraph, "grafoHashtags", True)

"""
#STANZA
#stanza.download('en', processors='tokenize,ner,pos') # se descarga el modelo en inglés y los procesadores necesarios
#nlp = stanza.Pipeline('en', processors='tokenize,ner,pos')  # se inicializa el Pipeline con el idioma y procesadores
#nlp = stanza.Pipeline('en', processors='tokenize,ner,pos', tokenize_pretokenized=True)  # initialize English neural pipeline
#nlp = stanza.Pipeline('en', processors='tokenize,pos')

#Concatenamos todos los tweets, ya que el rendimiento es mejor así en lugar de realizar un bucle for y realizar el análisis por cada tweet
df['Texto'] = df['Texto'].astype(str)
texto = '\n\n'.join(df['Texto'].to_list())

doc = nlp(texto)

# Gráfico de palabras
wordslist = extractwords(doc)

# Se ordena la lista de hashtags en función de mayor aparición y se recompone en 2 listas diferentes
sortedWords = sorted((zip(wordslist[1], wordslist[0])),reverse=True)
sortedNumberWords, sortedWords = zip(*sortedWords)

# Pintar gráficos de barras palabras
plotbarchart(10, sortedWords, sortedNumberWords, 'Top 10 palabras más utilizadas',
             'Palabra','Nº de veces','graficoPalabras')


# Gráfico de hashtags utilizados: Extrar hashtags y nº de veces de todos los tuits
hashtagList = re.findall('#(\w+)', texto)
hashtagList = [x.lower() for x in hashtagList]
hashtagList = [word for word in hashtagList if word not in stopwords]
hashtagList = np.unique(hashtagList, return_counts=True)

# Se ordena la lista de hashtags en función de mayor aparición y se recompone en 2 listas diferentes
sortedHashtags = sorted((zip(hashtagList[1], hashtagList[0])), reverse=True)
sortedNumberHashtags, sortedHashtags = zip(*sortedHashtags)

# Pintar gráficos de barras hashtags
plotbarchart(10, sortedHashtags, sortedNumberHashtags, 'Top 10 hashtags más utilizados',
             'Hashtag','Nº de veces','graficoHashtags')

"""
nlp = stanza.Pipeline('en', processors='tokenize,ner')
# FIN STANZA

"""
# GRAFICOS TEMPORALES

dfFechas = df['Fecha'].copy()
days = getDays(dfFechas)
plottemporalserie(days, dfMainHashtags, sortedMainHashtags,
                  "evolucionHashtagsUsados", "Evolución temporal de los hashtags más utilizados")
plottemporalserie(days, dfHashtagsRT, sortedHashtagsRT,
                  "evolucionHashtagsRT", "Evolución temporal de los hashtags más retuiteados")
"""


# COMUNIDADES

# Se descartan las componentes con menos de 4 nodos
toDeleteIds = []
clusters = hashtagsGraph.clusters()
for cluster in clusters:
    if len(cluster) < 4:
        toDeleteIds.extend(cluster)
hashtagsGraph.delete_vertices(toDeleteIds)

communities = plotcommunities(hashtagsGraph, "grafoHashtags", hashtagsGraph.clusters())



dfCommunities = df
tweets = dfCommunities.to_numpy()
communitiesArray = []

# Extrae la lista de hashtags por cada tweet y busca a qué comunidad pertenece
for idx, tweet in enumerate(tweets):
    matches = re.findall('#(\w+)', tweet[1].lower())
    try:
        matches = [word for word in matches if word not in stopwords]  # Si existe, se elimina de la lista el hashtag citizenscience
    except ValueError:
        pass
    communitiesInTweet = []
    for match in matches:
        i = 1
        for community in communities:
            if match in community:
                communitiesInTweet.append("Comunidad " + str(i))
            i += 1
    communitiesInTweet = np.unique(communitiesInTweet)
    communitiesArray.append(communitiesInTweet)

tweets = np.column_stack((tweets, communitiesArray))
dfCommunities = pd.DataFrame(tweets, columns=['Fecha','Texto','Usuario','Comunidades'])
dfCommunities.to_csv('data/comunidades.csv', header=True, index=False, sep=';')

communitiesList = []
i = 1
for community in communities:
    communitiesList.append("Comunidad " + str(i))
    i += 1

f = open("files/" + "InfoComunidades" + ".txt", "w+", encoding='utf-8')
j = 1
substring_list = ['project', 'program']
substring_list_nocount = ['citizenscience', 'citizen science', 'cienciaciudadana', 'ciencia ciudadana',
                          'communityscience', 'openscience', 'citsci','citizenscience and openscience',
                          'citizenscience & openscience']
#Recorrer tweets por comunidad
for community in communitiesList:
    dfCommunity = dfCommunities[pd.DataFrame(dfCommunities['Comunidades'].tolist()).isin([community]).any(1)]
    dfEliminarHashtagsRT2 = dfCommunity[dfCommunity['Texto'].str.match('RT:')]
    dfCommunity = dfCommunity.drop(dfEliminarHashtagsRT2.index, axis=0).copy()

    organizations = []
    people = []
    projects = []

    #Concatenamos texto de los tweets de una comunidad
    text = '\n\n'.join(dfCommunity['Texto'].to_list())

    doc = nlp(text)

    for sent in doc.sentences:
        # Entidades
        for entity in sent.entities:
            #if any(substring not in entity.text.lower() for substring in substring_list_nocount):
            textEntity = entity.text.replace("@", "")
            textEntity = textEntity.replace("#", "")

            if textEntity.lower() not in substring_list_nocount:
                if entity.type == 'ORG':
                    organizations.append(textEntity)
                elif entity.type == 'PERSON':
                    people.append(textEntity)

                if any(substring in textEntity.lower() for substring in substring_list):
                    projects.append(textEntity)

    organizations = np.unique(organizations)
    people = np.unique(people)
    projects = np.unique(projects)

    f.write(community + ":\n\n")
    f.write("Organizaciones: " + str(organizations) + "\n")
    f.write("Personas: " + str(people) + "\n")
    f.write("Proyectos: " + str(projects) + "\n")
    f.write("\n\n")

f.close()

print("Tiempo: " + str((time.time() - start_time) / 60))
