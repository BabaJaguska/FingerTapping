Sto se podataka tice, obradjeni su podaci sa desne ruke samo iako je
snimana i leva. Ovo je po Milicinoj instrukciji, ona smatra da je mnogo
smislenije fokusirati se samo na desnu ruku jer ove bolesti teze da se
manifestuju lateralizovano i to tako da je ovim pacijentima svima
dominantno desna ruka zahvacena. Mada nisam ni ja bas ljubitelj bacanja
podataka. 
splitovi se mogu automatski generisati iz ManuallyCheckTapSplits,
ako se ova skripta pozove sa flagom za auto, tj. 
ManuallyCheckTapSplits -d datapath_do_podataka -m 'auto'.  (smesta splitove u promenljivu data )
Posle bi se moglo pozvati sa -m 'file' da se koriguju auto splitovi i upisu
u fajl, ali ovo je bas dosadan i dug proces. (valjalo bi razdvojiti fje
upisivanja i proveravanja <-- ovo vise meni kao podsetnik)