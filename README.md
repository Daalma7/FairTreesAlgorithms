# Development of Fair Machine Learning Algorithms based on Decision Trees || Master's thesis

![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=white)
![Cython](https://img.shields.io/badge/cython-yellow.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1MTIiIGhlaWdodD0iNTEyIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiIgaW1hZ2UtcmVuZGVyaW5nPSJvcHRpbWl6ZVF1YWxpdHkiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZmlsbD0iI2VhZWJlYSIgeG1sbnM6dj0iaHR0cHM6Ly92ZWN0YS5pby9uYW5vIj48cGF0aCBkPSJNNTExLjUgMTQ3LjV2M2MtMzUuMzM1LjE2Ny03MC42NjggMC0xMDYtLjUtMjQuMjEzLTQ2LjEzMS02Mi4yMTMtNzMuMTMxLTExNC04MS00MC45MzEtNS41MTEtNzkuNTk4IDEuNDg5LTExNiAyMS0zMi40NDQgMjEuMTQtNTMuMjc4IDUwLjY0LTYyLjUgODguNS04LjgyMyA0Mi4xMjktOS44MjMgODQuNDYzLTMgMTI3IDcuMDM2IDQzLjkxMyAyNi41MzYgODEuMDc5IDU4LjUgMTExLjUgMjguMDc2IDIxLjk2OSA2MC4wNzYgMzIuNDY5IDk2IDMxLjUgMzguMTU3IDEuMzg3IDcyLjQ5MS05LjExMyAxMDMtMzEuNSAxMy4xNzMtMTEuMjA0IDIzLjg0LTI0LjM3IDMyLTM5LjVhMTcwMC41NiAxNzAwLjU2IDAgMCAxIDEwMS0uNWMtNDIuNTEzIDY3Ljk2NS0xMDMuODQ2IDEwNy4yOTgtMTg0IDExOC01NC4wMDYgNy41MTEtMTA3LjAwNiAzLjE3OC0xNTktMTNDODYuMTc3IDQ1NS4zNDQgMzcuNjc3IDQwNi4xNzcgMTIgMzM0LjVjLTYuMDUxLTIwLjA5LTEwLjIxNy00MC40MjMtMTIuNS02MXYtNDhjNS4wNzUtNTEuODEzIDI1LjU3NS05Ni40NzkgNjEuNS0xMzRDMTA4LjM3MyA0NC42OCAxNjUuNTQgMTguODQ3IDIzMi41IDE0YzU0LjQ4Mi01LjM2IDEwNy4xNDkgMS45NzQgMTU4IDIyIDUzLjMyNCAyMy4xNiA5My42NTcgNjAuMzI3IDEyMSAxMTEuNXoiIG9wYWNpdHk9Ii45ODMiLz48cGF0aCBkPSJNMjQ5LjUgMTAxLjVjMTQuMDA0LS4xNjcgMjguMDA0IDAgNDIgLjUgMTMuMjEyLjE3NCAyNS41NDUgMy41MDcgMzcgMTAgNy43NTYgNS42NjYgMTIuOTIyIDEzLjE2NiAxNS41IDIyLjUuNjY3IDI5IC42NjcgNTggMCA4Ny0zLjY2OCAxMy42NjctMTIuMTY4IDIyLjgzNC0yNS41IDI3LjVhMjE1MC4xOCAyMTUwLjE4IDAgMCAxLTkwIDJjLTIwLjIwMyAzLjg3MS0zMy4wMzcgMTUuNzA0LTM4LjUgMzUuNWEzNjQuMjcgMzY0LjI3IDAgMCAwLTEuNSA0M2MtNDAuNjA2IDcuNTQyLTYzLjEwNi05LjEyNS02Ny41LTUwLTQuNjY4LTI1LjY1Ny0yLjMzNC01MC42NTcgNy03NSA2LjI3Mi0xMi4yNjkgMTYuMTA1LTIwLjEwMiAyOS41LTIzLjUgMzcuNjMyLTEuNDYyIDc1LjI5OS0xLjk2MiAxMTMtMS41di04aC03M2E4NDAuMjcgODQwLjI3IDAgMCAxIC41LTQxYzIuMTM2LTEwLjEzOSA3Ljk2OS0xNy4zMDYgMTcuNS0yMS41IDExLjIyNS0zLjYwOSAyMi41NTgtNi4xMDkgMzQtNy41em0tMjMgMjNjMTIuMTUxLS4wMTUgMTcuMzE4IDUuOTg1IDE1LjUgMTgtMy44OTkgNy45NjktMTAuMDY2IDEwLjQ2OS0xOC41IDcuNS0xMC4wMTUtOS43MzYtOS4wMTUtMTguMjM2IDMtMjUuNXoiIG9wYWNpdHk9Ii45ODQiLz48cGF0aCBkPSJNMzUzLjUgMTc5LjVjMTEuMzM4LS4xNjcgMjIuNjcyIDAgMzQgLjUgMTMuNjA4IDMuNjA0IDIyLjc3NCAxMi4xMDQgMjcuNSAyNS41IDEzLjk3NSAzNy4yNzggMTEuOTc1IDczLjYxMS02IDEwOS00LjY3OSA3LjkxOS0xMS41MTIgMTIuNzUzLTIwLjUgMTQuNWwtMTE3IC41djloNzNjLjQyNSAxMy4zOTQtLjA3NSAyNi43MjctMS41IDQwLTMuNzc2IDguNzc2LTkuOTQyIDE1LjI3Ni0xOC41IDE5LjUtMzEuMDIgMTMuMzU3LTYyLjY4NyAxNS4wMjMtOTUgNS0xNS4zMTktNC4zMjYtMjUuODE5LTEzLjgyNi0zMS41LTI4LjUtLjY2Ny0yOC42NjctLjY2Ny01Ny4zMzMgMC04NiAzLjcwNy0xNC4yMzQgMTIuNTQtMjMuNzM0IDI2LjUtMjguNSAzMC4yOTktMS4yODYgNjAuNjMyLTEuOTUzIDkxLTIgMTkuMDg0LTQuNzUxIDMxLjI1LTE2LjU4NCAzNi41LTM1LjVhMzY0LjI3IDM2NC4yNyAwIDAgMCAxLjUtNDN6bS00NCAxNzhjMTIuODE1Ljk3MiAxNy42NDggNy42MzggMTQuNSAyMC01LjA2OCA3LjM2OC0xMS41NjggOC44NjgtMTkuNSA0LjUtNi4yMjktNi43MjUtNi41NjItMTMuNzI1LTEtMjEgMi4yMjctLjk0MSA0LjIyNy0yLjEwOCA2LTMuNXoiIG9wYWNpdHk9Ii45ODUiLz48L3N2Zz4=)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)
![Numpy](https://img.shields.io/badge/numpy-%23013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23285479.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPHN2ZyB2aWV3Qm94PSIwIDAgNTAwIDUwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcy8+CiAgPHBhdGggZD0iTSAyNDUuMzExIDYuMjc4IEMgNTguMjggOS4yNTIgLTU1LjQwMSAyMTMuNzk3IDQwLjY4NCAzNzQuNDYzIEMgMTM2Ljc3NCA1MzUuMTMgMzcwLjU2MyA1MzEuNDE1IDQ2MS41MDcgMzY3Ljc3NiBDIDU1MC43OTMgMjA3LjEyNCA0MzYuNjQ0IDkuMTk3IDI1My4wMjQgNi4yNzggTCAyNDUuMzExIDYuMjc4IFogTSAyNDUuMzExIDYwLjMzNiBDIDk5Ljg1NSA2My4zMDggMTIuMTU4IDIyMi44MDIgODcuNDU5IDM0Ny40MjMgQyAxNjIuNzU5IDQ3Mi4wNDUgMzQ0LjU3OCA0NjguMzI3IDQxNC43MzcgMzQwLjczNSBDIDQ2My4zNTQgMjUyLjMxMiA0MzMuMDczIDE0OS42NSAzNjAuNjI3IDk2LjQ0MiBDIDMzMC45ODIgNzQuNjY4IDI5NC4yNzUgNjEuMTc4IDI1My4wMjQgNjAuMzM2IEwgMjQ1LjMxMSA2MC4zMzYgWiBNIDI0NS4zMTEgMTE0LjM5MyBDIDE0MS40MzMgMTE3LjM2NiA3OS43MjYgMjMxLjc5OCAxMzQuMjM2IDMyMC4zNjggQyAxODguNzQ0IDQwOC45NDEgMzE4LjU4OSA0MDUuMjI2IDM2Ny45NTkgMzEzLjY4IEMgNDE1LjcwOCAyMjUuMTMzIDM1My41MDMgMTE3LjI2NSAyNTMuMDI0IDExNC4zOTMgTCAyNDUuMzExIDExNC4zOTMgWiBNIDI0NS4zMTEgMTY4LjQ1MSBDIDE4My4wMzEgMTcxLjQyNCAxNDcuMzE5IDI0MC43NzYgMTgxLjAzIDI5My4yODUgQyAyMTQuNzQzIDM0NS43OTQgMjkyLjU5MyAzNDIuMDc5IDMyMS4xNjIgMjg2LjU5NiBDIDM0OC4xOCAyMzQuMTI2IDMxMS45MjQgMTcxLjI2MSAyNTMuMDI0IDE2OC40NTEgTCAyNDUuMzExIDE2OC40NTEgWiBNIDI0NS4zMTEgMjIyLjUwOCBDIDIyNC43NDEgMjI1LjQ4MSAyMTUuMDk4IDI0OS42MjkgMjI3Ljk1NCAyNjUuOTc4IEMgMjQwLjgwOSAyODIuMzI2IDI2Ni41MjQgMjc4LjYxMiAyNzQuMjM3IDI1OS4yOSBDIDI4MC43MjUgMjQzLjA0NSAyNzAuMzE5IDIyNS4wMDYgMjUzLjAyNCAyMjIuNTA4IEwgMjQ1LjMxMSAyMjIuNTA4IFogTSA0OTIuMTU5IDI0OS41MzcgTCA2LjE3NiAyNDkuNTM3IE0gNDIyLjczMiA3NS43OCBMIDc1LjYwMyA0MjMuMjkzIE0gNDIyLjczMiA0MjMuMjkzIEwgNzUuNjAzIDc1Ljc4IE0gMjQ5LjE2NyA2LjI3OCBMIDI0OS4xNjcgNDkyLjc5NSIgc3R5bGU9InBhaW50LW9yZGVyOiBzdHJva2UgbWFya2VyczsgZmlsbC1ydWxlOiBldmVub2RkOyBmaWxsOiByZ2IoMjU1LCAyNTUsIDI1NSk7IGZpbGwtb3BhY2l0eTogMDsgc3Ryb2tlLXdpZHRoOiAxMHB4OyBzdHJva2U6IHJnYigyNTUsIDI1NSwgMjU1KTsiLz4KICA8cGF0aCBkPSJNIDE5NS4xNjkgMzMuMzA3IEwgMTE4LjAzIDcxLjkyIEwgMzgwLjMwNSA0MjcuMTU0IEwgNDE4Ljg3NSAzODguNTQyIEwgMjQ5LjE2NyAyNDkuNTM3IEwgMTk1LjE2OSAzMy4zMDcgWiIgc3R5bGU9ImZpbGw6IHJnYigyNTUsIDI1NSwgMjU1KTsiLz4KICA8cGF0aCBkPSJNIDY3Ljg4OCAxOTUuNDggTCA2Ny44ODggMzAzLjU5NCBMIDI0OS4xNjcgMjQ5LjUzNyBMIDY3Ljg4OCAxOTUuNDggWiIgc3R5bGU9ImZpbGw6IHJnYigyNTUsIDI1NSwgMjU1KTsiLz4KICA8cGF0aCBkPSJNIDI0OS4xNjcgMjQ5LjUzNyBMIDI3Mi4zMDkgMzg0LjY4IEwgMjI2LjAyNSAzODQuNjggTCAyNDkuMTY3IDI0OS41MzcgWiIgc3R5bGU9ImZpbGw6IHJnYigyNTUsIDI1NSwgMjU1KTsiLz4KICA8cGF0aCBkPSJNIDI0OS4xNjcgMjQ5LjUzNyBMIDI5OS4zMDkgOTUuMDg2IEwgMzM0LjAyMSAxMTQuMzkzIEwgMjQ5LjE2NyAyNDkuNTM3IFoiIHN0eWxlPSJmaWxsOiByZ2IoMjU1LCAyNTUsIDI1NSk7Ii8+CiAgPHBhdGggZD0iTSAyNDkuMTY3IDI0OS41MzcgTCAzMDMuMTY1IDIyNi4zNjkgTCAzMDcuMDIyIDI0MS44MTQgTCAyNDkuMTY3IDI0OS41MzcgWiBNIDI0OS4xNjcgMjQ5LjUzNyBMIDE0OC44ODUgMjk5LjczMyBMIDE2MC40NTcgMzE1LjE3OCBMIDI0OS4xNjcgMjQ5LjUzNyBaIiBzdHlsZT0iZmlsbDogcmdiKDI1NSwgMjU1LCAyNTUpOyIvPgo8L3N2Zz4=&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-%2388afbb.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MDAgNTAwIj48cGF0aCBkPSJNNiAyMDhjNy00NCAyNi04MiA1NC0xMTZBMjQyIDI0MiAwIDAgMSAyNzggNWEyMzkgMjM5IDAgMCAxIDE2MSA4NSAyNDggMjQ4IDAgMCAxIDU5IDE0OCAyNDYgMjQ2IDAgMCAxLTU2IDE2OGMtMjMgMjctNTEgNDgtODMgNjRBMjUxIDI1MSAwIDAgMSAyNyAzNTcgMjQ0IDI0NCAwIDAgMSA2IDIwOG0xNyAxMDNhMjM2IDIzNiAwIDAgMCAzNzcgMTE4Yzc3LTYxIDEwNi0xNjggNzAtMjU5QTIyNyAyMjcgMCAwIDAgMjc2IDE4IDIzMiAyMzIgMCAwIDAgMjMgMzEwWiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMSIvPjxwYXRoIGQ9Ik0yMyAzMTBBMjMzIDIzMyAwIDAgMSAyNzYgMThjOTUgMTIgMTYwIDYzIDE5NCAxNTEgMzYgOTEgNyAxOTgtNzAgMjYwQTIzNiAyMzYgMCAwIDEgMjMgMzEwbTQwNi0xODloMWwtMS0yLTEwLTEzYy02MC02NS0xMzUtOTAtMjIzLTcwQzEzNSA1MCA5MCA4NyA1NyAxNDBjLTcgMTItMTIgMjUtMTcgMzhsMiAyIDM1LTRjMjYtNiA1My05IDc3LTE5IDM1LTEzIDY5LTMwIDEwMy00NiA0NC0yMSA4OS0yNiAxMzYtNmwzNiAxNk0zNiAyODBsLTUgNCAxMCA0MiA1Mi0yM2MzNi0xNSA3My0xNiAxMDktMiAyMCA4IDM5IDE2IDU3IDI2IDUyIDI3IDEwNiA0NyAxNjUgNTEgNSAwIDgtMSAxMS01IDE4LTI4IDMwLTU4IDM1LTkwLTE3LTYtMzQtMTAtNTAtMTctMzEtMTMtNjEtMjgtOTItNDMtNTEtMjUtMTAzLTI2LTE1NCAwbC00MiAyMWMtMzEgMTUtNjIgMjktOTYgMzZtMTQyIDI2Yy00OS0xMC05MSA4LTEzMyAzMCAzOSA3OSAxMDAgMTI2IDE4OCAxMzMgNzYgNiAxMzgtMjIgMTg4LTc4LTI1LTUtNDktOS03My0xNi0zNy05LTcwLTI4LTEwNC00NGwtNjYtMjVtLTM0LTgwIDM1LTE3YzQ0LTIxIDg5LTI0IDEzNS0zbDUzIDI1YzI5IDE0IDU4IDI3IDg5IDM1bDE2IDRjMy00MS00LTc5LTIxLTExNS01LTEwLTExLTE4LTIyLTIybC0zMC0xNGMtNDAtMTctODAtMjEtMTIxLTRsLTY2IDMwYy01MyAyOC0xMTAgNDQtMTcwIDQ3LTUgMC03IDItOCA3bC02IDM3djM0YzQyLTggNzktMjYgMTE2LTQ0WiIgZmlsbD0idHJhbnNwYXJlbnQiIG9wYWNpdHk9IjEiLz48cGF0aCBkPSJNMzYgMjgwYzM0LTcgNjUtMjEgOTYtMzZsNDItMjFjNTEtMjYgMTAzLTI1IDE1NCAwIDMxIDE1IDYxIDMwIDkyIDQzIDE2IDcgMzMgMTEgNTAgMTctNSAzMi0xNyA2Mi0zNSA5MC0zIDQtNiA1LTExIDUtNTktNC0xMTMtMjQtMTY1LTUxLTE4LTEwLTM3LTE4LTU3LTI2LTM2LTE0LTczLTEzLTEwOSAybC01MiAyMy0xMC00MiA1LTRaIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIxIi8+PHBhdGggZD0ibTE3OSAzMDYgNjUgMjVjMzQgMTYgNjcgMzUgMTA0IDQ0IDI0IDcgNDggMTEgNzMgMTYtNTAgNTYtMTEyIDg0LTE4OCA3OC04OC03LTE0OS01NC0xODgtMTMzIDQyLTIyIDg0LTQwIDEzNC0zMFptLTM2LTgwYy0zNiAxOC03MyAzNi0xMTUgNDR2LTM0bDYtMzdjMS01IDMtNyA4LTcgNjAtMyAxMTctMTkgMTcwLTQ3bDY2LTMwYzQxLTE3IDgxLTEzIDEyMSA0bDMwIDE0YzExIDQgMTcgMTIgMjIgMjIgMTcgMzYgMjQgNzQgMjEgMTE1bC0xNi00Yy0zMS04LTYwLTIxLTg5LTM1bC01My0yNWMtNDYtMjEtOTEtMTgtMTM1IDNsLTM2IDE3WiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMSIvPjxwYXRoIGQ9Im00MjkgMTIwLTM2LTE1Yy00Ny0yMC05Mi0xNS0xMzYgNi0zNCAxNi02OCAzMy0xMDMgNDYtMjQgMTAtNTEgMTMtNzcgMTlsLTM1IDQtMi0yYzUtMTMgMTAtMjYgMTctMzggMzMtNTMgNzgtOTAgMTM5LTEwNCA4OC0yMCAxNjMgNSAyMjMgNzBsMTAgMTRaIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIxIi8+PHBhdGggZD0ibTQyOSAxMjAgMSAxaC0xdi0xWiIgZmlsbD0iI0IzRDZERSIgb3BhY2l0eT0iMSIvPjwvc3ZnPg==&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?logo=plotly&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?logo=github&logoColor=white)

![DataScience](https://img.shields.io/badge/Data_Science-finished?color=283D59)
![ML](https://img.shields.io/badge/ML-finished?color=595828)
![AI](https://img.shields.io/badge/AI-finished?color=594628)
![FML](https://img.shields.io/badge/Fairness_in_ML-finished?color=1C2835)
![MOO](https://img.shields.io/badge/MOO-finished?color=282C59)
![EAs](https://img.shields.io/badge/EAs-finished?color=555555)
![Probability](https://img.shields.io/badge/Probability-finished?color=285958)
![Analysis](https://img.shields.io/badge/Analysis-finished?color=131313)


![Status](https://img.shields.io/badge/status-finished-Green)
![Grade](https://img.shields.io/badge/grade-High_Honors-yellow)
![License](https://img.shields.io/badge/license-MIT-red)



This repository contains the work done on the implementation and testing of three different fair  machine learning algorithms based on decision trees for binary classification with one binary sensitive attribute, and multiobjective evolutionary procedures using this algorithms to achieve the best balance between accuracy and fairness. You can read the project report in the file [MasterThesis.pdf](MasterThesis.pdf).

## Abstract

Fairness in machine learning is a field that has gained great prominence and relevance in recent years. This area deals with the study and quantification of different measures of fairness on various specific problems and decision processes, as well as the development and implementation of solutions to create fairer systems. Unfortunately, at the limits of joint optimization between accuracy and fairness, a trade-off is reached, so that requiring a fairer system will necessarily imply less accuracy and vice versa. Due to this fact, multiobjective optimization emerges as a method that allows finding a wide range of solutions exploring this optimization frontier. Genetic algorithms represent the state of the art in these optimization processes.

This project will be developed within this area. After conducting a review of the context from a mathematical perspective, analyzing various interpretations of mathematical fairness, and exploring the theory behind multi-objective optimization, the creation of 3 novel algorithms based on decision trees for fair binary classification with a binary sensitive attribute will be proposed. The first algorithm, named Fair Decision Tree (FDT), modifies the information gain criterion during decision tree training to incorporate fairness. The second, named Fair Genetic Pruning (FGP), proposes a genetic optimization procedure on prunings in a matrix tree, which will be the tree that perfectly classifies the training sample. The last one, named Fair LightGBM (FLGBM), modifies the loss function of the LightGBM algorithm to incorporate fairness. For the two algorithms that do not inherently use genetic optimization processes, a genetic hyperparameter optimization procedure based on the NSGA-II algorithm will be employed. This will enable finding models on the joint optimization frontier, specifically for accuracy and fairness objectives. A study will be conducted to test these algorithms alongside a baseline algorithm (classic decision tree) on 10 relevant datasets within the field. Each dataset will be split into 10 training, validation, and test partitions using different random seeds. The average results obtained demonstrate how these algorithms are capable of finding a wide range of Pareto-optimal solutions. Overall, FDT and FLGBM algorithms outperform the baseline algorithm across several quality indicators calculated on the average Pareto fronts. The FGP algorithm has also shown promising results but has not surpassed the baseline algorithm on these indicators. However, it is the algorithm that achieved the best processing time results. All the algorithms have found models that perform better on both objectives than classic models like the COMPAS model developed by Northpointe.

## Brief descrition of developed algorithms
- **FairDT (FDT)**: Modification of the impurity criterion calculation during decision tree training to also consider fairness. Its general expression is: $$(1-\lambda) \cdot k \cdot \text{gini/entropy} - \lambda \cdot \text{fairness criterion}$$ Where $\lambda$ is the hyperparameter that controls the importance given to the fairness criterion used. k is a normalization factor, with $k=2$ if the Gini impurity criterion is used, and $k=1$ is the entropy impurity criterion is used, since the fairness criterion will take values in the range $[0,1]$. Different strategies are proposed for calculating the value of the fairness criterion at any internal node, as a classification is needed to compute it. Finally, a method named "Fair probabilistic prediction" is used.

- **Fair Genetic Pruning (FGP)**: Genetic pruning of a matrix decision tree (the largest decision tree that can be built to perfectly classify the training sample, considering different balances for both classes). This algorithm is an evolutionary algorithm, where each individual in the population is represented by the codes of the nodes where the prunings occur. These codes are lexicographically ordered for efficiency, avoiding pruning already pruned branches of the tree. Crossover assigns prunings from parents to children one by one, ensuring that each pruning assigned is performed in a non-pruned node. Mutations select a leaf of the tree and traverse up or down the matrix tree structure by a certain number of random levels, which depend on the matrix tree depth, then apply or substitute the pruning. This process is repeated a random number of times, depending on the number of leaves of the individual.

- **FairLGBM (FLGBM)**: Modification of the loss function in the LightGBM algorithm to incorporate fairness. $$(1-\lambda) \cdot k \cdot \text{logloss} + \lambda \cdot \text{continuous fairness criterion}$$ In fact, it has the following real structure: $`L_f(Y, \Sigma,P) = k(1-\lambda) L(Y,\Sigma) +\lambda\left| \frac{𝟙_{-,0,1}^T \Sigma}{𝟙_{-,0,1}^T 1} -\frac{𝟙_{-,0,0}^T \Sigma}{𝟙_{-,0,0}^T 1} \right|`$ Where $L$ is the logloss function, $\Sigma$ are the prediction scores (after applying the logistic function) and $`\left| \frac{𝟙_{-,0,1}^T \Sigma}{𝟙_{-,0,1}^T 1} -\frac{𝟙_{-,0,0}^T \Sigma}{𝟙_{-,0,0}^T 1} \right|`$ is the continuous extension of $\text{FPR}_{\text{diff}}$ fairness criterion. This continuous extension is highly advantageous for the calculation of the derivative and hessian of this function, which is necessary for its implementation in a LightGBM algorithm. $k=\frac{-1}{\ln{0.5}}$ is a "normalization factor", considered using as the worst prediction the one made by a dummy classifier ($\Sigma=(0.5,0.5,\dots, 0.5$)). This is not the worst prediction, but it serves as a good normalization point. 

## Brief description of the experimentation
The experimentation involved testing each algorithm with 10 datasets for binary classification that are well-known in the field of fairness in machine learning (adult, compas, diabetes, dutch, german, insurance, obesity, parkinson, ricci, and student), containing 1 binary protected attribute. To obtain robust and reproducible results, each experiment was run a total of 10 times with different values of the random seed (from 100 to 109), which controls the partitioning of data into training, validation, and test sets, as well as other pseudo-random processes. Average results were calculated from the outcomes obtained in each run for each algorithm. The experimentation was conducted with the three developed algorithms, as well as with a decision tree (DT).

The hyperparameters that define the decision space of each algorithm are as follows:

- **DT**:
    - **criterion**: gini / entropy.
    - **max_depth**: maximum depth of the tree.
    - **min_samples_split**: minimum number of samples required to split an internal node.
    - **max_leaf_nodes**: maximum number of leaf nodes the final tree can have.
    - **class_weight**: weight assigned to each class to be predicted.
- **FDT**:
    - **same parameters**, plus:
    - **fair_param**: parameter that controls the balance between the impurity criterion and the fairness criterion during tree learning.
- **FGP**
    - The method itself is a genetic algorithm that returns a large number of solutions. Instead of hyperparameter optimization of a base classifier, this method is applied directly.
- **FLGBM**
    - **num_leaves**: number of leaves in the tree.
    - **min_data_in_leaf**: minimum amount of data required in a leaf node to allow splitting.
    - **max_depth**: maximum depth of the tree.
    - **learning_rate**: learning rate of the algorithm.
    - **n_estimators**: number of weak classifiers to be built.
    - **feature_fraction**:  fraction of features used to build the model.
    - **class_weight**: weight assigned to each class to be predicted (internally uses scale_pos_weight).
    - **fair_param**: controls the balance between the standard loss function (logloss) of the algorithm and the fairness function considered.

The objectives to minimize during the experimentation are as follows:

- **Inverted G-mean** (gmean_inv): The geometric mean criterion is defined as the square root of the product of the true positive rate and the true negative rate $(\sqrt{\text{TPR} \cdot \text{TNR}})$. Since it is a minimization objective, $1-\sqrt{\text{TPR} \cdot \text{TNR}}$ will be used.
- **Difference in False Positive Rate** (fpr_diff):  This is the difference between the probabilities $\text{FPR}_{\text{diff}} = |P[p=1|Y=0,A=0]-P[p=1|Y=0,A=1]|$, where $p$ is the result of the classifier, $Y$ is the attribute to be predicted, and $A$ is the sensitive attribute.

An overall structure of the experimentation conducted for each dataset can be seen in this Figure. The population size used was $n_i=150$, while the number of generations used was $n_g=300$. The probability of crossover $p_c=1$ (as both previous and current population will be joined applying elitist selection) and the probability of mutation $p_M=0.3$:
![An overall structure of the experimentation conducted for each dataset can be seen in this Figure. The population size used was $n_i=150$, while the number of generations used was $n_g=300$. The probability of crossover $p_c=1$ (as both previous and current population will be joined applying elitist selection) and the probability of mutation $p_M=0.3$.](other/METHOD.png)


## Main results

The results show that FDT and FLGBM algorithms generally achieve better average Pareto fronts with respect to almost all quality indicators compared to the base DT algorithm. Although the FGP algorithm did not improve upon these quality indicators relative to the base algorithm, it still produced very good models. Additionally, it achieved the best results in terms of CPU processing time. The models found demonstrate significant improvement over commonly used classic models, such as the COMPAS model from Northpointe.

<div align="center">
    <img src="other/scatter_po_algorithm_adult.png" width="412px"/> 
    <img src="other/scatter_po_algorithm_compas.png" width="412px"/> 
</div>
<div align="center">
    <img src="other/scatter_po_algorithm_diabetes.png" width="412px"/> 
    <img src="other/scatter_po_algorithm_dutch.png" width="412px"/> 
</div>
<div align="center">
    <img src="other/scatter_po_algorithm_german.png" width="412px"/> 
    <img src="other/scatter_po_algorithm_insurance.png" width="412px"/> 
</div>
<div align="center">
    <img src="other/scatter_po_algorithm_obesity.png" width="412px"/> 
    <img src="other/scatter_po_algorithm_parkinson.png" width="412px"/> 
</div>
<div align="center">
    <img src="other/scatter_po_algorithm_ricci.png" width="412px"/> 
    <img src="other/scatter_po_algorithm_student.png" width="412px"/> 
</div>

The rankings of the quality measures on the average Pareto fronts found can also be consulted here:

<div align="center">
    <img src="other/ranking_Hypervolume.png" width="412px"/> 
    <img src="other/ranking_Maximum Spread.png" width="412px"/> 
</div>
<div align="center">
    <img src="other/ranking_Spacing.png" width="412px"/> 
    <img src="other/ranking_Overall Pareto Front Spread.png" width="412px"/> 
</div>
<div align="center">
    <img src="other/ranking_Error Ratio.png" width="412px"/> 
    <img src="other/ranking_Generational Distance.png" width="412px"/> 
</div>
<div align="center">
    <img src="other/ranking_Inverted Generational Distance.png" width="412px"/> 
</div>

The average CPU processing times of each generation (in seconds) can also be consulted here:
![Average execution times of each generation (in seconds)](other/execution_times.jpg)

Overall, these good results show that there is ample room for improvement in the construction of fair classification models for joint optimization of accuracy and fairness.

## Libraries and dependencies:

Listed below are the libraries and dependencies required to run your own experiments. You can try using higher versions of all libraries except Cython:
- **python**=3.10.12
- **matplotlib**=3.8.3
- **pandas**=2.2.1
- **scikit-learn**=1.4.1.post1
- **pydotplus**=2.0.2
- **imblearn**
- **cython**=0.29.37
- **lightgbm**=4.3.0 (from the official lightgbm webpage)
- **seaborn**=0.13.2
- **pygmo**=2.19.5
<!-- conda create --name NAME conda-forge python=3.10.12 -->
<!-- conda activate NAME -->
<!-- pip install matplotlib -->
<!-- pip install pandas -->
<!-- pip install scikit-learn -->
<!-- pip install pydotplus -->
<!-- pip install imblearn -->
<!-- pip install cython=0.29.37 -->
<!-- execute build.sh inside /HyperparameterOptimization/models/FairDT -->
<!-- install lightgbm with cuda support from the lightgbm webpage -->
<!-- pip install seaborn -->
<!-- pip install pygmo -->

--- 

## Additional info
- **Author**: David Villar Martos
- **Contributors**: David Villar Martos
- **Project director**: Jorge Casillas Barranquero

