import nltk

base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')]

# stopwords manuais
stopwords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou']

# selecionando stopwords da biblioteca nltk
stopwordsNltk = nltk.corpus.stopwords.words("portuguese")

# Remover as stopwords do texto e retirar os radicais dessas palavras
def aplicaStemmer(texto):

    # ltk.stem.RSLPStemmer() utilizado para trabalhar com a linguagem portuguesa
    stemmer = nltk.stem.RSLPStemmer()
    frasesStemmer = []
    for (palavras, emocao) in texto:
        comStemming = [str(stemmer.stem(p)) for p in palavras.split() if not p in stopwordsNltk]
        frasesStemmer.append((comStemming, emocao))
    return frasesStemmer


# remover as stopwords do texto
def removeStopwords(texto):
    frases = []
    for (palavras, sentimento) in texto:
        semStopword = [p for p in palavras.split() if p not in stopwordsNltk]
        frases.append((semStopword, sentimento))
    return frases


# Retorna todas as palavras
def buscarPalavras(frases):
    todasPalavras = []
    for (palavras, emocao) in frases:
        todasPalavras.extend(palavras)
    return todasPalavras

# retorna as frequencias das palavras
def buscarFrequencia(palavras):
    palavras =  nltk.FreqDist(palavras)
    return palavras

# Retorna as palavras unicas no text
def buscarPalavrasUnicas(frequencia):
    return frequencia.keys()

# Extrair as palavras
def extratorPalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasUnicas:
        caracteristicas["%s" %palavras] = (palavras in doc)
    return caracteristicas


#print(removeStopwords(base))

palavrasStemmer =  aplicaStemmer(base)
#print(palavrasStemmer)

palavras = buscarPalavras(palavrasStemmer)
#print(palavras)

frequencia = buscarFrequencia(palavras)
#print(frequencia)

palavrasUnicas = buscarPalavrasUnicas(frequencia)
#print(palavrasUnicas)

caracteristicas = extratorPalavras(["am", "nov", "dia"])
#print(caracteristicas)


baseCompleta = nltk.classify.apply_features(extratorPalavras, palavrasStemmer)
#print(baseCompleta)

# Constroi  a tabela de probabilidade
# cria classificador de teste
classificador = nltk.NaiveBayesClassifier.train(baseCompleta)

# Imprimir as labels existentes
#print(classificador.labels())

print(classificador.show_most_informative_features(5))