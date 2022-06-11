library(ggplot2)
library("ggpubr")
require(gridExtra)
require(dplyr)
data_path <- "/Users/zeehondje/Documents/GitHub/thesis/"


file <- "dvs–11-rnn-snntorch"
file <- "dvs–11-cnn-snntorch"
file <- "dvscnntmsetemp"
file <- "–mnist-cnn-binds"


create_plot <- function(path, title){
  trainfile <- paste(data_path,"train",path,".csv", sep = "")
  testfile <- paste(data_path,"test",path,".csv", sep = "")
  
  df_train <- read.csv(trainfile)
  df_test <- read.csv(testfile)
  df_train$Phase <- "train"
  df_test$Phase <- "test"
  df <- rbind(df_train, df_test)
  
  
  ## making the accuracy graph
  accuracy <- df %>%group_by(Phase, Epoch)%>% summarise(avg = mean(Accuracy),sd = sd(Accuracy))
  accuracy$se <- accuracy$sd
  accuracy$se[accuracy$Phase=="train"] <- (accuracy$sd[accuracy$Phase=="train"]/sqrt(133))
  accuracy$se[accuracy$Phase=="test"] <- (accuracy$sd[accuracy$Phase=="train"]/sqrt(30))
  
  top3 <- df %>%group_by(Phase, Epoch)%>% summarise(avg = mean(Top3),sd = sd(Top3))
  top3$se <- (top3$sd/sqrt(133))
  top3$se <- top3$sd
  top3$se[top3$Phase=="train"] <- (top3$sd[top3$Phase=="train"]/sqrt(133))
  top3$se[top3$Phase=="test"] <- (top3$sd[top3$Phase=="train"]/sqrt(30))
  
  accuracy$Top <- "1"
  top3$Top <- "3"
  sumy <- rbind(accuracy, top3)
  
  
  ggplot(sumy, aes(x=Epoch, y=avg, color=Top, linetype = Phase)) + 
    geom_errorbar(aes(ymin= avg-se, ymax= avg+se), width=.1) +
    geom_line() +
    geom_point() +
    ylim(c(0,1)) +
    ylab("Accuracy") +
    scale_x_continuous(breaks = seq(0, 10, len = 11))+
    labs(title = title)+
    scale_color_manual(values=c("green", "blue"))
  
  save_file <- paste("Documents/GitHub/thesis/", path, ".eps", sep = "")
  ggsave(file = save_file, device = "eps", width=14, height = 8, units="cm")
}


create_plot("Mnist1RnnCeB04", 
            "Learning MNIST with surrogate gradients \nRNN with no hidden layers \nLoss: CE, neuron decay = 0.4")
create_plot("Mnist1RnnMSEB04", 
            "Learning MNIST with surrogate gradients \nRNN with no hidden layers \nLoss: MSE, neuron decay = 0.4")
create_plot("Mnist2RnnCeB04", 
            "Learning MNIST with surrogate gradients \nRNN with one hidden layers \nLoss: CE, neuron decay = 0.4")
create_plot("Mnist2RnnMseB04", 
            "Learning MNIST with surrogate gradients \nRNN with one hidden layers \nLoss: MSE, neuron decay = 0.4")
create_plot("MnistCnnCeB04", 
            "Learning MNIST with surrogate gradients \nCNN \nLoss: CE, neuron decay = 0.4")
create_plot("MnistCnnMseB04", 
            "Learning MNIST with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.4")
create_plot("MnistCnnMseB04", 
            "Learning MNIST with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.4")

create_plot("Mnist1RnnCeB02", 
            "Learning MNIST with surrogate gradients \nRNN with no hidden layers \nLoss: CE, neuron decay = 0.2")
create_plot("Mnist1RnnMSEB02", 
            "Learning MNIST with surrogate gradients \nRNN with no hidden layers \nLoss: MSE, neuron decay = 0.2")
create_plot("Mnist2RnnCeB02", 
            "Learning MNIST with surrogate gradients \nRNN with one hidden layers \nLoss: CE, neuron decay = 0.2")
create_plot("Mnist2RnnMseB02", 
            "Learning MNIST with surrogate gradients \nRNN with one hidden layers \nLoss: MSE, neuron decay = 0.2")
create_plot("MnistCnnCeB02", 
            "Learning MNIST with surrogate gradients \nCNN \nLoss: CE, neuron decay = 0.2")
create_plot("MnistCnnMseB02", 
            "Learning MNIST with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.2")
create_plot("MnistCnnMseB02", 
            "Learning MNIST with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.2")

create_plot("Dvs1RnnCeB04", 
            "Learning DVS Gesture with surrogate gradients \nRNN with no hidden layers \nLoss: CE, neuron decay = 0.4")
create_plot("Dvs1RnnMSEB04", 
            "Learning DVS Gesture with surrogate gradients \nRNN with no hidden layers \nLoss: MSE, neuron decay = 0.4")
create_plot("Dvs2RnnCeB04", 
            "Learning DVS Gesture with surrogate gradients \nRNN with one hidden layers \nLoss: CE, neuron decay = 0.4")
create_plot("Dvs2RnnMseB04", 
            "Learning DVS Gesture with surrogate gradients \nRNN with one hidden layers \nLoss: MSE, neuron decay = 0.4")
create_plot("DvsCnnCeB04", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: CE, neuron decay = 0.4")
create_plot("DvsCnnMseB04", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.4")
create_plot("DvsCnnMseB04", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.4")

create_plot("Dvs1RnnCeB02", 
            "Learning DVS Gesture with surrogate gradients \nRNN with no hidden layers \nLoss: CE, neuron decay = 0.2")
create_plot("Dvs1RnnMSEB02", 
            "Learning DVS Gesture with surrogate gradients \nRNN with no hidden layers \nLoss: MSE, neuron decay = 0.2")
create_plot("Dvs2RnnCeB02", 
            "Learning DVS Gesture with surrogate gradients \nRNN with one hidden layers \nLoss: CE, neuron decay = 0.2")
create_plot("Dvs2RnnMseB02", 
            "Learning DVS Gesture with surrogate gradients \nRNN with one hidden layers \nLoss: MSE, neuron decay = 0.2")
create_plot("DvsCnnCeB02", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: CE, neuron decay = 0.2")
create_plot("DvsCnnMseB02", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.2")
create_plot("DvsCnnMseB02", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.2")

create_plot("dvscnntmsetemp", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: MSE, neuron decay = 0.2")


create_plot("DvsCnnMseB04F24_2", 
            "Learning DVS Gesture with surrogate gradients \nCNN \nLoss: CE, neuron decay = 0.4")




# Export ggplot2 plot
## making the Loss graph
loss <- df %>%group_by(Phase, Epoch)%>% summarise(avg = mean(Loss),sd = sd(Loss))
loss$se <- (loss$sd/sqrt(133))

ggplot(loss, aes(x=Epoch, y=avg, linetype = Phase)) + 
  geom_errorbar(aes(ymin= avg-se, ymax= avg+se), width=.1) +
  geom_line() +
  geom_point() +
  ylab("Loss") +
  scale_x_continuous(breaks = seq(0, 10, len = 11))
  



  
