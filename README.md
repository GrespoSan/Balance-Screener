# G. Balance Stock Screener V4.3

Correzione esclusivamente sulla freschezza dati Daily italiani.

- Yahoo/yfinance resta la sorgente principale.
- Retry Yahoo individuale se la Daily è arretrata.
- Per ticker .MI ancora arretrati: fallback pubblico Euronext sulle sole sedute recenti mancanti.
- Se anche Euronext non aggiorna il dato, il ticker viene escluso dallo screening invece di usare una Daily vecchia.
- Nessuna modifica a motore Balance, dominanza S/R, AREA ATTIVA, tocchi, Score o filtri.
