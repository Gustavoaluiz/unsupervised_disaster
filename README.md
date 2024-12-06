# unsupervised_disaster
Trabalho final referente à disciplina de Aprendizado de Máquina Não Supervisionado do Bacharelado em Inteligência Artificial.

# Dados
https://drive.google.com/file/d/1on4_cbLyDDS3sXyllwpk7qzVwoZQg3kJ/view?usp=sharing

# Introdução

# Problema
* O problema foi inspirado na competição "https://xview2.org/", onde a motivação está no fato em que informações rápidas e precisas são essenciais para uma resposta eficaz à desastres. Para isso, é necessário saber a área afetada, a causa e a gravidade dos danos - nisso, imagens de satélite são uma excelente forma de fazer essa comunicação. No entanto, a imagem por si só não tem tanto valor, sendo necessária realizar uma anotaçõo manual para poderem ser comunicadas - e é nisso que a competição se baseia - anotação automatizada dessas imagens para deixar a assistência humanitária mais eficiente.
* Partindo disso, limitamos o escopo da competição para ser compatível com o escopo e tempo de trabalho para a disciplina, onde selecionamos somente a parte do problema que se refere a descobrir que tipo de desastre natural afetou certa área.
* A relevância dessa detecção está tanto na rápida resposta, quanto em decidir que tipo de socorrista deve ser enviado para tomar as devidas ações.
* Além disso, percebemos que a disponibilização de imagens de qualidade desse tipo não é algo tão fácil, recorrente e abundante, mesmo porque não estão acontecendo desastres de todos os tipos o tempo todo. Portanto, para contribuir ainda mais com o escopo do trabalho, pensamos em usar esse problema para testar a hipótese de que gerar imagens desse tipo para aumentar a quantidade e variabilidade do dataset pode contribuir para a performance de modelos detectadores desses desastres.

# Objetivos
* Desenvolvedor dois classificadores: um para detectar se a área realmente foi afetada por algum desastre ou não, e outro para dado que a área realmente foi afetada, descobrir qual desastre aconteceu.
* Testar a hipótese de que a geração de imagens para aumento da quantidade e variabilidade dos dados pode melhorar a perfomance dos modelos.

# Metodologia

## Classificação com Rede Neural

## Data Augmentation com Stable Diffusion

* Realização do fine-tuning "Textual Inversion" em um modelo ramificação do Stable Diffusion encontrado no HuggingFace, no intuito de realizar um Data Augmentation em imagens áreas fotografadas por um satélite de desastres naturais, com o objetivo de melhorar um classificador inicial que retorna se a imagem aérea é comum, ou representa um desastre.
* Baseline: https://colab.research.google.com/drive/16kRKFcuF5eqlkiMfE_gundYy9opB8C3Q?usp=sharing

## Classificação utilizando imagens geradas

# Conclusões