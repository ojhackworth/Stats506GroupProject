##ques3a 
library(dplyr)
library(tidyverse)
recs15<-read.csv('./recs2015_public_v3.csv')
recs15<-tbl_df(recs15)                    

a11<-group_by(recs15,DIVISION,WALLTYPE)               
a12<-summarise(a11,homesum=sum(NWEIGHT))                  ##sum follow the parts in division
hometperct<-mutate(a12,perct=100*homesum/118208250)       ##home total percent, 118208250=sum(homesum)

a13<-gather(recs15,key='brrWT',value='w',BRRWT1:BRRWT96)   ##change long to wide,545856*645
brrwt_homet<-select(a13,DIVISION,WALLTYPE,brrWT,w)

a14<-group_by(brrwt_homet,DIVISION,brrWT,WALLTYPE)
a15<-summarise(a14,homesun_r=sum(w))
a16<-group_by(a15,DIVISION,brrWT)
brrwt_homet_dist<-mutate(a16,perct_r=100*homesun_r/118208250)
a17<-ungroup(brrwt_homet_dist)                                   ##very important
a18<-left_join(a17, hometperct, by=c('DIVISION','WALLTYPE'))     ##join the nweight and brrwt

a19<-group_by(a18,DIVISION,WALLTYPE)
a110<-summarize(a19,perct=perct[1],
                std_error=2*sqrt(mean({perct_r-perct}^2)))
homet_whole<-mutate(a110,upr=perct+qnorm(0.975)*std_error,              
             lwr=perct-qnorm(0.975)*std_error)                      ## std_error, lwr, upr in 95% CI
q3a<-filter(homet_whole, WALLTYPE==4)                             ##filter



##ques3b

a21<-select(recs15,DIVISION,NWEIGHT,KWH)
a22<-group_by(a21,DIVISION)
ave_usage<-summarise(a22,mean=sum(NWEIGHT*KWH)/118208250)         ##sum by division

a23<-gather(recs15,key='brrWT',value='w',BRRWT1:BRRWT96)          ##change long to wide, by BRRWT
a24<-group_by(a23,DIVISION,brrWT)
a25<-summarise(a24,mean_r=sum(w*KWH)/sum(w))
ave_usage_wt<-left_join(a25,ave_usage,by=c('DIVISION'))           ##very important
a26<-group_by(ave_usage_wt,DIVISION)                              
a27<-summarize(a26,mean=mean[1],
               std_error=2*sqrt(mean({mean_r-mean}^2)))
ave_usage_whole<-mutate(a27,upr=mean+std_error*qnorm(0.975),
                        lwr=mean-std_error*qnorm(0.975))               ##electricity usage in each division

recs15_urban<-mutate(recs15, area=ifelse(recs15$UATYP10=='R','Rural','Urban'))  ##define the new data frame

a28<-select(recs15_urban,DIVISION,NWEIGHT,KWH,area)                ##exactly the same step with previous
a29<-group_by(a28,DIVISION,area)
ave_usage2<-summarise(a29,mean=sum(NWEIGHT*KWH)/118208250)
a210<-gather(recs15_urban,key='brrWT',value='w',BRRWT1:BRRWT96)
a211<-group_by(a210,DIVISION,area,brrWT)
a212<-summarise(a211,mean_r=sum(w*KWH)/118208250)
ave_usage_wt2<-left_join(a212,ave_usage2,by=c('DIVISION','area'))

a213<-group_by(ave_usage_wt2,DIVISION,area)
a214<-summarize(a213,mean=mean[1],
                std_error=2*sqrt(mean({mean_r-mean}^2)))
q3b<-mutate(a214,upr=mean+std_error*qnorm(0.975),
                         lwr=mean-std_error*qnorm(0.975))

##ques3c

a31<-select(recs15_urban,DIVISION,area,INTERNET,NWEIGHT)                ##exactly the same with previous
a32<-group_by(a31,DIVISION,area)
a33<-summarise(a32,perct=100*sum((INTERNET==1)*NWEIGHT)/118208250)
div_disp<-summarise(a33,disp=abs(perct[1]-perct[2]))
a34<-gather(recs15_urban,key='brrWT',value='w',BRRWT1:BRRWT96)
a35<-group_by(a34,DIVISION,brrWT,area)
a36<-summarise(a35,perct=100*sum(w*(INTERNET==1))/118208250)
a37<-summarise(a36,disp_r=abs(perct[1]-perct[2]))
a38<-left_join(a37,div_disp,by=c('DIVISION'))

a39<-group_by(a38,DIVISION)
a310<-summarise(a39,disp=disp[1],
                std_error=2*sqrt(mean({disp_r-disp}^2)))
div_disp_whole<-mutate(a310,upr=disp+std_error*qnorm(0.975),
                       lwr=disp-std_error*qnorm(0.975))
q3c<-arrange(div_disp_whole,disp)





































