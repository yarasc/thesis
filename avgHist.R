library(ggplot2)
library("ggpubr")
require(gridExtra)
require(dplyr)
data_path <- "/Users/zeehondje/Documents/GitHub/thesis/"



sum_hist <- function(path){
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
  accuracy$se[accuracy$Phase=="train"] <- (accuracy$sd[accuracy$Phase=="train"]/max(df_train$Iteration))
  accuracy$se[accuracy$Phase=="test"] <- (accuracy$sd[accuracy$Phase=="train"]/max(df_test$Iteration))
  
  top3 <- df %>%group_by(Phase, Epoch)%>% summarise(avg = mean(Top3),sd = sd(Top3))
  top3$se <- (top3$sd/max(df_train$Iteration))
  top3$se <- top3$sd
  top3$se[top3$Phase=="train"] <- (top3$sd[top3$Phase=="train"]/max(df_train$Iteration))
  top3$se[top3$Phase=="test"] <- (top3$sd[top3$Phase=="train"]/max(df_test$Iteration))
  
  accuracy$Top <- "1"
  top3$Top <- "3"
  sumy <- rbind(accuracy, top3)
  return(sumy)
}

sumy0<-sum_hist("10Dvs2RnnMseB04_0")
sumy0$run <- 0
sumy1<-sum_hist("10Dvs2RnnMseB04_1")
sumy1$run <- 1
sumy2<-sum_hist("10Dvs2RnnMseB04_2")
sumy2$run <- 2
sumy3<-sum_hist("10Dvs2RnnMseB04_3")
sumy3$run <- 3

sumy <- rbind(sumy0, sumy1, sumy2, sumy3)
sumy <- sumy %>%group_by(Phase, Epoch, Top,)%>% summarise(m = mean(avg),sd = sd(avg))
sumy$se <- sumy$sd
sumy$se[sumy$Phase=="train"] <- (sumy$sd[sumy$Phase=="train"]/max(df_train$Iteration))
sumy$se[sumy$Phase=="test"] <- (sumy$sd[sumy$Phase=="train"]/max(df_test$Iteration))

title = "Learning DVS Gesture with surrogate gradients \nRNN with one hidden layers \nLoss: MSE, neuron decay = 0.2"
ggplot(sumy, aes(x=Epoch, y=m, color=Top, linetype = Phase)) + 
  geom_errorbar(aes(ymin= m-se, ymax= m+se), width=.1) +
  geom_line() +
  geom_point() +
  ylim(c(0,1)) +
  ylab("Accuracy") +
  scale_x_continuous(breaks = seq(0, 10, len = 11))+
  labs(title = title)+
  scale_color_manual(values=c("green", "blue"))

test <- filter(sumy, Phase=="test", Top ==1)
max(sumy$m[sumy$Phase=="test", sumy$Top==1])

save_file <- paste("Documents/GitHub/thesis/", path, ".eps", sep = "")
ggsave(file = save_file, device = "eps", width=14, height = 8, units="cm")

