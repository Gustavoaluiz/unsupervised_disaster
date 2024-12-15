# unsupervised_disaster
Trabalho final referente à disciplina de Aprendizado de Máquina Não Supervisionado do Bacharelado em Inteligência Artificial.

# Dados
https://drive.google.com/file/d/1on4_cbLyDDS3sXyllwpk7qzVwoZQg3kJ/view?usp=sharing

# Introdução

# Problema
* O problema foi inspirado na competição "https://xview2.org/", onde a motivação está no fato de que informações rápidas e precisas são essenciais para uma resposta eficaz a desastres. Para isso, é necessário saber a área afetada, a causa e a gravidade dos danos - nisso, imagens de satélite são uma excelente forma de fazer essa comunicação. No entanto, a imagem por si só não tem tanto valor, sendo necessário realizar uma anotação manual para poderem ser comunicadas - e é nisso que a competição se baseia - anotação automatizada dessas imagens para deixar a assistência humanitária mais eficiente.
* Partindo disso, limitamos o escopo da competição para ser compatível com o escopo e tempo de trabalho para a disciplina, onde selecionamos somente a parte do problema que se refere a descobrir que tipo de desastre natural afetou certa área.
* A relevância dessa detecção está tanto na rápida resposta, quanto em decidir que tipo de socorrista deve ser enviado para tomar as devidas ações.
* Além disso, percebemos que a disponibilização de imagens de qualidade desse tipo não é algo tão fácil, recorrente e abundante, mesmo porque não estão acontecendo desastres de todos os tipos o tempo todo. Portanto, para contribuir ainda mais com o escopo do trabalho, pensamos em usar esse problema para testar a hipótese de que gerar imagens desse tipo para aumentar a quantidade e variabilidade do dataset pode contribuir para a performance de modelos detectores desses desastres.

# Objetivos
* Desenvolver dois classificadores: um para detectar se a área realmente foi afetada por algum desastre ou não, e outro para, dado que a área realmente foi afetada, descobrir qual desastre aconteceu.
* Testar a hipótese de que a geração de imagens para aumento da quantidade e variabilidade dos dados pode melhorar a performance dos modelos.

# Metodologia

## Classificação com Rede Neural

Para a construção da baseline do classificador, utilizamos uma arquitetura baseada na ResNet, implementada manualmente para garantir flexibilidade e personalização ao projeto.

### Estrutura da Baseline
Nossa implementação consiste nos seguintes componentes principais:

1. **Bloco Residual (ResidualBlock)**:
    - Cada bloco residual possui duas camadas convolucionais seguidas por normalização em lote (*Batch Normalization*) e ativação ReLU.
    - Conforme a teoria lançada pela ResNet, implementamos atalhos (skip connections) que adicionam diretamente a entrada do bloco à saída convolucional, na ideia de evitar o desaparecimento do gradiente ao longo das layers. Quando há diferenças de dimensões entre a entrada e a saída, utilizamos uma Conv2d 1x1 para ajustá-las corretamente.

2. **Camadas da ResNet**:
    - A rede começa com uma camada convolucional inicial que aumenta a profundidade das imagens para 64 canais, seguida por uma Batch Normalization, uma ReLU e uma camada de max pooling para reduzir as dimensões espaciais.
    - Quatro blocos residuais consecutivos são utilizados, com o número de map features aumentando progressivamente (64, 128, 128 e 256 canais). Com excessão do primeiro bloco, os seguintes utilizam "stride=2" para realizar a redução de dimensionalidade conforme as ativações passam pelas layers.

3. **Camadas Finais**:
    - Após os blocos residuais, uma camada de global average pooling reduz a saída para um único vetor de características por canal.
    - A saída é então passada por uma MLP, que produz a classificação final.

### Motivação da Escolha da ResNet

A ResNet foi escolhida devido a implementação de blocos residuais, que ainda fazem parte de arquiteturas amplamente utilizadas nas redes mais modernas de visão computacional, uma vez que conseguiram mitigar muito bem o problema de desaparecimento de gradiente. Além disso, a arquitetura da rede é robusta por si só e, por isso, tentamos aplicar o mais fiel possível à implementação original, mas adaptada ao nosso poder computacional e particularidades do nosso problema.

O intuito da baseline é gerar métricas iniciais em relação ao nosso problema para, posteriormente, podermos comparar o impacto da geração de imagens como data augumentation.

## Data Augmentation com Stable Diffusion

Para aumentar a quantidade e a variabilidade de dados no nosso dataset, utilizamos técnicas de geração de imagens baseadas no Stable Diffusion, um modelo de difusão conhecido por sua capacidade de gerar imagens de alta qualidade a partir de descrições textuais (prompts).

### Experimentos Iniciais
Inicialmente, testamos diversos modelos de Stable Diffusion com o objetivo de gerar imagens representativas dos cenários de desastres naturais necessários. Como esperado, não foi possível obter resultados satisfatórios apenas utilizando *prompts* genéricos. Entretanto, experimentos com diferentes modelos e ajustes nos *prompts* geraram imagens que eram próximas do necessário, mas ainda não plenamente adequadas. Assim, identificamos a necessidade de realizar um ajuste fino no modelo.

### Ajuste Fino com Textual Inversion
Devido às limitações de poder computacional e à escassez de dados disponíveis, optamos por utilizar a técnica de *Textual Inversion*. Segundo Gal et al. (2022), essa abordagem é uma solução eficiente para ajustar modelos de difusão pré-treinados, permitindo gerar imagens específicas sem a necessidade de re-treinar o modelo completo.

#### Funcionamento
- A técnica consiste em otimizar novos embeddings textuais para que representem conceitos específicos baseados em um pequeno conjunto de imagens (geralmente 3-5). Esses embeddings são inseridos no espaço de palavras do modelo, permitindo que novos conceitos sejam combinados com prompts existentes.
- O modelo aprende a associar o conceito com um pseudo-word, que pode ser utilizado para gerar imagens específicas e ajustadas às nossas necessidades.

### Resultados
Nossa hipótese inicial era que, com um ajuste fino eficiente, seria possível adaptar o modelo para gerar imagens representativas. Os resultados confirmaram essa hipótese, demonstrando que o Textual Inversion é uma abordagem viável para enriquecer datasets em cenários de baixa disponibilidade de dados.

### Relevância e Embasamento
A escolha do Textual Inversion é suportada pelos benefícios claros de eficiência computacional e pela alta fidelidade de representação dos conceitos. Além disso, sua integração no pipeline de data augmentation se alinha com os objetivos do projeto ao possibilitar a criação de imagens específicas para cenários de desastres naturais.

## Classificação utilizando imagens geradas

Após a etapa de geração de imagens, utilizamos as imagens geradas para treinar classificadores adicionais, avaliando seu impacto na performance geral dos modelos.

# Conclusões
O trabalho mostrou que a integração de abordagens de data augmentation com técnicas de difusão, como o Textual Inversion, pode ser uma solução eficaz para enriquecer datasets escassos, melhorando a performance em tarefas de classificação de desastres naturais.
