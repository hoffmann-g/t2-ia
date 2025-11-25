```
Pontifícia Universidade Católica - PUCRS
```
```
Disciplina: Inteligência Artificial
Prof. Lucas Rafael Costella Pessutto
```
```
Trabalho 2
```
# Estratégias de Aprendizagem em Prompts

Objetivo: O objetivo deste trabalho é estudar estratégias de prompting que permitam ensinar
um LLM a aprender a executar uma determinada tarefa.

Administrativo: O trabalho deve obedecer as seguintes regras:

- Grupos: o trabalho deve ser feito em **duplas, trios ou quartetos** , não sendo possível
    realizar individualmente.
- Data de Entrega: 17/11/2025 (Turma 31) | 25/11/2025 (Turma 33), até às 23:59, via
    moodle. Apenas um dos integrantes deve submeter os arquivos do trabalho no moodle.
- Apresentações: Aulas dos dias 18/11/2025 e 25/11/2025 (Turma 31) | 26/11/
    (Turma 33). Todos os integrantes devem estar presentes e devem participar da apre-
    sentação.
- Escolha de Grupos: Haverá uma atividade no moodle onde deverão ser indicados os
    grupos e as tarefas escolhidas. A escolha da data e do horário de apresentação é por
    ordem de indicação.
- Avaliação: A nota deste trabalho será dividida em: Execução (70%) e Apresentação
    (30%).

A tarefa
Neste trabalho vamos explorar duas técnicas de aprendizagem de máquina que se populari-
zaram recentemente, devido ao uso de prompting: **few-shot learning** e **zero-shot learning**.
Para isso, cada grupo deverá definir uma tarefa de aprendizagem de máquina. É esperado
que os grupos usem a criatividade para propor tarefas que sejam interessantes e úteis. Após
isso, deve ser construído um dataset de testes, que será utilizado para avaliar o desempenho
do LLM. O dataset deve ser inédito, criado pelo grupo, e seu tamanho deve ser 10 vezes o
número de integrantes do grupo (por exemplo, um trio deve criar um dataset de 30 exemplos).
A próxima fase do trabalho está em criar um prompt que instrua um LLM sobre como realizar
a tarefa. O prompt deve ser criado sem exemplos (zero-shot) e com exemplos (few-shot). Este
tutorial mostra como esse tipo de prompt pode ser criado: https://www.promptingguide.ai/
pt.

## 1


Em seguida devem ser realizados testes no prompting escolhido de modo a avaliar compara-
tivamente o desempenho do LLM nestas tarefas. O ideal é realizar estes testes via chamada
de API. Alguns LLMs como o Gemini fornecem créditos gratuitos para realizar esse tipo de
testes.
Cada grupo deve ainda propor um experimento adicional, que permita realizar comparações
adicionais. Pode-se testar uma técnica diferente de prompting, um LLM diferente ou até
abordagens clássicas de aprendizado de máquina sobre esses dados.
Por fim, o grupo deve produzir um relatório, em formato de artigo da SBC^1. Esse relatório
deve conter as seguintes seções:

1. Introdução: Apresentação inicial da tarefa e motivação.
2. Solução Desenvolvida: Descrever os prompts criados e as técnicas de prompting utiliza-
    das
3. Experimentos: Descrição dos experimentos: LLMs utilizados, configurações de parâme-
    tros dos LLMs, etc.
4. Resultados: Resultados obtidos nos experimentos e análise crítica desses resultados.
5. Conclusão: Conclusão do trabalho.

Apresentação
A apresentação do trabalho será feita para toda a turma nas aulas previstas no cronograma.
Cada grupo terá em torno de 7 minutos para se apresentar. Deverão ser confeccionados alguns
slides para a apresentação. Todos os membros do grupo devem estar presentes e devem par-
ticipar da apresentação.

Entrega
Devem ser entregues no repositório do moodle:

- o dataset criado pelo grupo
- todo o código desenvolvido para esta tarefa
- relatório no formato pdf
- apresentação de slides do seminário

(^1) Você consegue obter os templates da SBC no site: https://www.sbc.org.br/documentosinstitucionais/
#publicacoes. Há também um template para LATEXdisponível no Overleaf: https://pt.overleaf.com/latex/
templates/sbc-conferences-template/blbxwjwzdngr
2


