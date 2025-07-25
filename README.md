Cloned Hidden Markov Model

chmm.py --- содержит класс CHMM

mpg.py --- генератор последовательностей символов, соответствующих марковскому процессу, представленному на рисунке. 
Ноды данной сети представляют собой скрытые состояния процесса, стрелочки --- возможные переходы между ними. Цифры около стрелок указывают вероятность перехода, 
а буквы --- генерируемые символы в результате данного перехода. Таким образом, данный марковский процесс задаёт некоторую грамматику.

hmm_runner.py --- содержит класс HMMRunner, запускающий CHMM на последовательностях, сгенерированных в ходе марковского процесса. 
Здача модели --- научиться минимизировать неожиданность входных наблюдений. В ходе обучения вычисляются метрики качества: 
неожиданность ($-\log P(o)$), расстояние Кульбака-Лейблера и строится сглаженный портрет предсказываемых распределений в каждом состоянии марковского процесса.
