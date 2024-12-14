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

Para a construção da baseline do classificador, utilizamos uma arquitetura baseada na ResNet, implementada manualmente para garantir flexibilidade e personalização ao projeto. Essa rede foi desenvolvida com base em blocos residuais, uma abordagem amplamente utilizada e bem-sucedida para lidar com problemas de degradação em redes profundas.

### Estrutura da Baseline
Nossa implementação consiste nos seguintes componentes principais:

1. **Bloco Residual (ResidualBlock)**:
    - Cada bloco residual possui duas camadas convolucionais seguidas por normalização em lote (*Batch Normalization*) e ativação ReLU.
    - Para permitir que os gradientes fluam mais facilmente durante o treinamento, utilizamos atalhos (skip connections) que adicionam diretamente a entrada do bloco à saída convolucional. Quando há diferenças de dimensões entre a entrada e a saída, um atalho ajusta essas dimensões utilizando uma convolução de 1x1.

2. **Camadas da ResNet**:
    - A rede começa com uma camada convolucional inicial que aumenta a profundidade das imagens para 64 canais, seguida por uma normalização em lote, uma ativação ReLU e uma camada de *max pooling* para reduzir as dimensões espaciais.
    - Quatro blocos residuais consecutivos são utilizados, com o número de filtros aumentando progressivamente (64, 128, 128 e 256 canais). Para aumentar a hierarquia da abstração das características, a redução das dimensões espaciais é realizada utilizando *stride* em alguns blocos.

3. **Camadas Finais**:
    - Após os blocos residuais, uma camada de *global average pooling* reduz a saída para um único vetor de características por canal.
    - A saída é então passada por uma camada totalmente conectada (*fully connected layer*), que produz a classificação final.

### Motivação da Escolha da ResNet
A ResNet foi escolhida devido à sua robustez e capacidade de treinar redes profundas sem sofrer com o problema de vanishing gradients, graças ao uso de conexões residuais. Essa abordagem é particularmente importante em tarefas de classificação onde os padrões relevantes podem ser complexos e demandar redes profundas para sua detecção.

A baseline servirá como ponto de partida para avaliações futuras e será comparada a outras abordagens, incluindo aquelas que utilizam técnicas de *data augmentation* com imagens geradas.

## Data Augmentation com Stable Diffusion

Para aumentar a quantidade e a variabilidade de dados no nosso dataset, utilizamos técnicas de geração de imagens baseadas no Stable Diffusion, um modelo de difusão conhecido por sua capacidade de gerar imagens de alta qualidade a partir de descrições textuais (*prompts*).

### Experimentos Iniciais
Inicialmente, testamos diversos modelos de Stable Diffusion com o objetivo de gerar imagens representativas dos cenários de desastres naturais necessários. Como esperado, não foi possível obter resultados satisfatórios apenas utilizando *prompts* genéricos. Entretanto, experimentos com diferentes modelos e ajustes nos *prompts* geraram imagens que eram próximas do necessário, mas ainda não plenamente adequadas. Assim, identificamos a necessidade de realizar um ajuste fino no modelo.

### Ajuste Fino com Textual Inversion
Devido às limitações de poder computacional e à escassez de dados disponíveis, optamos por utilizar a técnica de *Textual Inversion*. Segundo Gal et al. (2022), essa abordagem é uma solução eficiente para ajustar modelos de difusão pré-treinados, permitindo gerar imagens específicas sem a necessidade de re-treinar o modelo completo.

#### Funcionamento
- A técnica consiste em otimizar novos *embeddings* textuais para que representem conceitos específicos baseados em um pequeno conjunto de imagens (geralmente 3-5). Esses *embeddings* são inseridos no espaço de palavras do modelo, permitindo que novos conceitos sejam combinados com *prompts* existentes.
- O modelo aprende a associar o conceito com um *pseudo-word*, que pode ser utilizado para gerar imagens específicas e ajustadas às nossas necessidades.

### Resultados
Nossa hipótese inicial era que, com um ajuste fino eficiente, seria possível adaptar o modelo para gerar imagens representativas. Os resultados confirmaram essa hipótese, demonstrando que o *Textual Inversion* é uma abordagem viável para enriquecer datasets em cenários de baixa disponibilidade de dados&#8203;:contentReference[oaicite:1]{index=1}.

### Relevância e Embasamento
A escolha do *Textual Inversion* é suportada pelos benefícios claros de eficiência computacional e pela alta fidelidade de representação dos conceitos&#8203;:contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}. Além disso, sua integração no pipeline de *data augmentation* se alinha com os objetivos do projeto ao possibilitar a criação de imagens específicas para cenários de desastres naturais.

## Classificação utilizando imagens geradas

Após a etapa de geração de imagens, utilizamos as imagens geradas para treinar classificadores adicionais, avaliando seu impacto na performance geral dos modelos.

# Conclusões
O trabalho mostrou que a integração de abordagens de *data augmentation* com técnicas de difusão, como o *Textual Inversion*, pode ser uma solução eficaz para enriquecer datasets escassos, melhorando a performance em tarefas de classificação de desastres naturais.
