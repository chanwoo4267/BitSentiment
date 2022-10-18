import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Coin from './components/features/coin/Coin';

import Logo from './assets/logo.svg';

//api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false

function App() {
  const [coins, setCoins] = useState([]);
  const [search, setSearch] = useState('');
  const [fng, setFng] = useState(0);

  useEffect(() => {
    axios
      .get(
        'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=30&page=1&sparkline=false'
      )
      .then((res) => {
        setCoins(res.data);
        console.log(res.data);
      })
      .catch((error) => console.log(error));
  }, []);

  const [analyzed, setAnalyzed] = useState({});
  useEffect(() => {
    axios
      .get('http://112.186.88.149:8000/api/main')
      .then((res) => {
        setAnalyzed(res.data['coin']);
        setFng(res.data['fng']);
      })
      .catch((error) => console.log(error));
  }, []);

  const handleChange = (e) => {
    setSearch(e.target.value);
  };

  const filteredCoins = coins.filter((coin) =>
    coin.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className='coin-app'>
      <a href='/' className='logo'>
        <img src={Logo} width='100px' height='100px' alt='' />
      </a>
      <h1 className='bitsentiment-logo'>BitSentiment</h1>
      <div className='coin-search'>
        <form>
          <input
            className='coin-input'
            type='text'
            onChange={handleChange}
            placeholder='Search'
          />
        </form>
      </div>
      <div className='fng'>
        <h2>Market Fear & Greed: {fng.toFixed(2)}</h2>
      </div>
      <div className='header-container'>
        <div className='header-row'>
          <h3 className='header-rank'>Rank</h3>
          <div className='header'>
            <h3>Coin</h3>
            <h3 className='header-symbol'>Symbol</h3>
          </div>
          <div className='header-data'>
            <h3 className='header-price'>Price</h3>
            <h3 className='header-volume'>Total Volume</h3>
            <h3 className='header-percent'>24h Change</h3>
            <h3 className='header-sentiment'>BitSentiment Score</h3>
            <h3 className='header-marketcap'>Market Cap</h3>
          </div>
        </div>
        {filteredCoins.map((coin, index) => {
          return (
            <Coin
              key={coin.id}
              rank={coins.indexOf(coin) + 1}
              name={coin.name}
              price={coin.current_price}
              symbol={coin.symbol}
              marketcap={coin.market_cap}
              volume={coin.total_volume}
              image={coin.image}
              priceChange={coin.price_change_percentage_24h}
              sentiment={analyzed[coin.symbol.toUpperCase()]}
            />
          );
        })}
      </div>
    </div>
  );
}

export default App;
