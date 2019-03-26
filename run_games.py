import subprocess


games_whole = ['Hopper-v2', 'HalfCheetah-v2', 'Swimmer-v2', 'Ant-v2', 'Walker2d-v2', 'Reacher-v2']
games_test = ['Hopper-v2','Ant-v2', 'Walker2d-v2', 'Reacher-v2']
games = games_whole

for game in range(len(games)):
    # d = i % 2
    # cmd = "CUDA_VISIBLE_DEVICES=%d python run_erl.py -env %s &" % (d, games[i])
    cmd = "python main.py --env_name %s --node_name qcis5 &" % games[i]
    returned_value = subprocess.call(cmd, shell=True)
    # print('returned value:', returned_value)