
from sys import stderr
from Logger import *
from conv_policy_value import *
from Gomoku import *
import random

def main():

    vp = ValuePolicy()

    model_name = 'model_05.pt'
    processed_data_path = "datas\\botzone_data\\processed_data\\Gomoku-2021-5\\"

    import datetime
    current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    sys.stdout = Logger('logs//'+current_time+'.log',sys.stdout)
    sys.stderr = Logger('logs//'+current_time+'.log',sys.stderr)

    total_epoches = 20
    batch_size = 1
    train_test_split = 0.1
    game_number = 200
    total_games = [i for i in range(game_number)]
    test_games = total_games[:int(game_number*train_test_split)]
    train_games = total_games[int(game_number*train_test_split):]
    
    print('\nStart training model: "{}" with data from: "{}"\n'.format(model_name,processed_data_path))

    for e in range(total_epoches):
        random.shuffle(train_games)
        print(">>>>training epoches :",e+1)
        count = 0

        r_policy_loss = 0
        r_value_loss = 0
        for game_num in train_games:
            print("  trained ",count * 100,"games.")
            count += 1

            if count % 30 == 0:
                vp.save_model(model_name)
                print("model saved! to {}".format(model_name))

            start = game_num * 100 + 1
            end = start + 100 - 1
            file_name = str(start) + "-" + str(end)
            states_sets = np.load(processed_data_path+file_name+"_states_sets.npy")
            act_probs =  np.load(processed_data_path+file_name+"_act_probs.npy")
            values = np.load(processed_data_path+file_name+"_values.npy")

            policy_loss = 0
            value_loss = 0
            for i in range(int(states_sets.shape[0]/batch_size)):
                c_policy_loss,c_value_loss = vp.train(states_sets[i*batch_size:(i+1)*batch_size,:,:,:],act_probs[i*batch_size:(i+1)*batch_size,:],values[i*batch_size:(i+1)*batch_size])
                policy_loss += c_policy_loss
                value_loss += c_value_loss
            r_policy_loss += (policy_loss / int(states_sets.shape[0]/batch_size))
            r_value_loss += (value_loss / int(states_sets.shape[0]/batch_size))

        print(" >>loss this epoch on train set: policy ",round(r_policy_loss/len(train_games),3)," value ",round(r_value_loss/len(train_games),3))


        t_policy_loss = 0
        t_value_loss = 0
        for game_num in test_games:
            start = game_num * 100 + 1
            end = start + 100 - 1
            file_name = str(start) + "-" + str(end)
            states_sets = np.load(processed_data_path+file_name+"_states_sets.npy")
            act_probs =  np.load(processed_data_path+file_name+"_act_probs.npy")
            values = np.load(processed_data_path+file_name+"_values.npy")
            policy_loss = 0
            value_loss = 0
            for i in range(int(states_sets.shape[0]/batch_size)):
                c_policy_loss,c_value_loss = vp.test(states_sets[i*batch_size:(i+1)*batch_size,:,:,:],act_probs[i*batch_size:(i+1)*batch_size,:],values[i*batch_size:(i+1)*batch_size])
                policy_loss += c_policy_loss
                value_loss += c_value_loss
            t_policy_loss += (policy_loss / int(states_sets.shape[0]/batch_size))
            t_value_loss += (value_loss / int(states_sets.shape[0]/batch_size))

        print(" >>loss this epoch on test set: policy ",round(t_policy_loss/len(test_games),3)," value ",round(t_value_loss/len(test_games),3))

        vp.save_model('epoch_'+str(e)+'_'+model_name)
        print("One epoch finished! model saved! to {}".format(model_name))


if __name__ == "__main__":
    main()
    
