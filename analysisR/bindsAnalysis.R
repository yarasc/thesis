
path <- "train–mnist-eth-binds"
testfile <- paste(data_path,path,".csv", sep = "")

df<- read.csv(testfile)


## making the accuracy graph
accuracy <- df %>%group_by(Epoch)%>% summarise(avg = mean(Accuracy),sd = sd(Accuracy))
accuracy$se <- (accuracy$sd/max(df_test$Iteration))

top3 <- df %>%group_by(Epoch)%>% summarise(avg = mean(Top3),sd = sd(Top3))
top3$se <- top3$sd/max(df_train$Iteration)
accuracy$Top <- "1"
top3$Top <- "3"
sumy <- rbind(accuracy, top3)



ggplot(sumy, aes(x=Epoch, y=avg, color=Top)) + 
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

