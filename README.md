# Super Mario Bros
This project was my first general learning of neural networks and how they work within a program. I have done some NN work in the past but never anything
above a "textbook example", so this project really let me discover how to wrap code around the neuron and hidden layer.

## How it runs
- 20 marios run in parallel during each generations. during run time all 20 are displayed in a grid with the current best performing mario being displayed largly on the top right
- Each Mario runs through the relatively small feedforward neural network, this network is 208 inputs with a hidden layer holding 18 nodes and holds 12 outputs. Each output is mapped to a different NES input; jump, left, right, etc.
- Instead of mario memorizing a set amount of moves like string evolution I wanted it to be able to know what its doing. Each input is based off the NES Ram map built into the game, being able to find walls, marios position and enemies. This was pulled from a website for Super Mario Bros RAM map
- For the fitness I wanted to give mario some goals that equated to a better score, such as coin count, speed, getting as far as it could and a penalty for death.
- Each generation ends and the population breeds, a small number of elite marios get brought to the new generation unchanged, and the other marios in population will breed as well as a small mutation chance for random evolution
- The best NN at the end of the population collection is saved to disk memory after each generation

### Running the Program 
- At launch it will create some files, and a folder. Pycache is created to hold python bytecode
- best_neural_network.npz holds the best NN from the most recent generation
- last_population is the full population, generation count and fitness history for the fitness display graph

## Training
The script is designed to be able to stop and resume as the operator needs, each generation is saved when it reaches completion. A check is done within code to allow to resume from a current .npz file but if none is detected then a new random population will be created and run from there. 
Its safe to stop the simulation when ever with Control+C (on windows), this will not corrupt anything or lose any training progress 

## Known Issues
- Currently the level is hard set on World 4 level 1 and is hard coded in EM_Ram.py
- If there is not enough progress at the start and the fitness level is below 10 then the mating pool will be empty and will not be able to make a new generation. This has not happened to me during testing however the math is technically there and it could happen on first run, very rare
- Currently the program hits a spot during the training where it can not get past a big gap and needs to "long jump", it has done that in simulation before however currently after 20 generations it still walks off the edge. From what I have read this is a known issue with genetic algorithms and hitting a local optimum. Looking into a fix

Project originally came to thought through wondering what evolutionary system I could make, then I found a public github repo [Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros)
to learn how to frame the github emulator to work on my computer, then using OpenCv to run the window. Without this repo the project would not have been posible for me

## To Run
to install all dependencies please run 'pip install -r requirements.txt' to get all the packages. See ['requirements.txt'](requirements.txt) for the exact package list.

Then run:
'python main.py'. 
At launch this will fire up the emulator and create some files, 'best_neural_network.npz', 'last_population.npz' and '__pycache__'
