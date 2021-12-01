from a_little_complex_policy_value import *
from Gomoku import *
import random

def main():

    vp = ValuePolicy()

    vp.load_model("save/saved_models/high_abc_70000/abc_policy_value_e_2.pt")
    print("load save/saved_models/high_abc_70000/abc_policy_value_e_2.pt successfully!")

    total_epoches = 20
    batch_size = 1
    train_test_split = 0.5
    game_number = 300
    total_games = [i for i in range(game_number)]
    test_games = total_games[:int(game_number*train_test_split)]
    train_games = total_games[int(game_number*train_test_split):]
    
    for e in range(2,total_epoches):
        random.shuffle(train_games)
        print(">>>>training epoches :",e+1)
        count = 0

        r_policy_loss = 0
        r_value_loss = 0
        for game_num in train_games:
            print("  trained ",count * 100,"games.")
            count += 1

            if count % 30 == 0:
                vp.save_model("save/saved_models/high_abc_70000/abc_policy_value_e_"+str(e+1)+".pt")
                print("model saved! to save/saved_models/high_abc_70000/abc_policy_value_e_"+str(e+1)+".pt")

            start = game_num * 100 + 1
            end = start + 100 - 1
            file_name = str(start) + "-" + str(end)
            states_sets = np.load("datasets\\high_level_data_08\\"+file_name+"_states_sets.npy")
            act_probs =  np.load("datasets\\high_level_data_08\\"+file_name+"_act_probs.npy")
            values = np.load("datasets\\high_level_data_08\\"+file_name+"_values.npy")

            policy_loss = 0
            value_loss = 0
            for i in range(int(states_sets.shape[0]/batch_size)):
                c_policy_loss,c_value_loss = vp.train(states_sets[i*batch_size:(i+1)*batch_size,:,:,:],act_probs[i*batch_size:(i+1)*batch_size,:],values[i*batch_size:(i+1)*batch_size])
                policy_loss += c_policy_loss
                value_loss += c_value_loss
            r_policy_loss += policy_loss / int(states_sets.shape[0]/batch_size)
            r_value_loss += value_loss / int(states_sets.shape[0]/batch_size)

        print(" >>loss this epoch on train set: policy ",round(r_policy_loss/len(train_games),3)," value ",round(r_value_loss/len(train_games),3))


        t_policy_loss = 0
        t_value_loss = 0
        for game_num in test_games:
            start = game_num * 100 + 1
            end = start + 100 - 1
            file_name = str(start) + "-" + str(end)
            states_sets = np.load("datasets\\high_level_data_08\\"+file_name+"_states_sets.npy")
            act_probs =  np.load("datasets\\high_level_data_08\\"+file_name+"_act_probs.npy")
            values = np.load("datasets\\high_level_data_08\\"+file_name+"_values.npy")
            policy_loss = 0
            value_loss = 0
            for i in range(int(states_sets.shape[0]/batch_size)):
                c_policy_loss,c_value_loss = vp.test(states_sets[i*batch_size:(i+1)*batch_size,:,:,:],act_probs[i*batch_size:(i+1)*batch_size,:],values[i*batch_size:(i+1)*batch_size])
                policy_loss += c_policy_loss
                value_loss += c_value_loss
            t_policy_loss += policy_loss / int(states_sets.shape[0]/batch_size)
            t_value_loss += value_loss / int(states_sets.shape[0]/batch_size)

        print(" >>loss this epoch on test set: ",round(t_policy_loss/len(test_games),3),round(t_value_loss/len(test_games),3))

        vp.save_model("save/saved_models/high_abc_70000/abc_policy_value_e_"+str(e+1)+".pt")
        print("model saved ! to save/saved_models/high_abc_70000/abc_policy_value_e_"+str(e+1)+".pt")        


if __name__ == "__main__":
    main()
    
