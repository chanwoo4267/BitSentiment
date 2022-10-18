import React from 'react';
import './Coin.css';

import SemiCircleProgressBar from 'react-progressbar-semicircle';

const Coin = ({
  rank,
  name,
  price,
  symbol,
  marketcap,
  volume,
  image,
  priceChange,
  sentiment = 0,
}) => {
  return (
    <div className='coin-container'>
      <div className='coin-row'>
        <p className='coin-rank'>#{rank}</p>
        <div className='coin'>
          <img src={image} alt='crypto' />
          <h1>{name}</h1>
          <p className='coin-symbol'>{symbol}</p>
        </div>
        <div className='coin-data'>
          <p className='coin-price'>${price}</p>
          <p className='coin-volume'>${volume.toLocaleString()}</p>

          {priceChange < 0 ? (
            <p className='coin-percent red'>{priceChange.toFixed(2)}%</p>
          ) : (
            <p className='coin-percent green'>{priceChange.toFixed(2)}%</p>
          )}
          <div className='coin-sentiment'>
            {sentiment ? (
              sentiment < 50 ? (
                <p className='sentiment red'>
                  <SemiCircleProgressBar
                    percentage={sentiment.toFixed(2)}
                    diameter={100}
                    showPercentValue
                    strokeWidth={5}
                    stroke='#f00606'
                  />
                </p>
              ) : (
                <p className='sentiment green'>
                  <SemiCircleProgressBar
                    percentage={sentiment.toFixed(2)}
                    diameter={100}
                    showPercentValue
                    strokeWidth={5}
                    stroke='#11d811'
                  />
                </p>
              )
            ) : (
              '-'
            )}
          </div>
          <p className='coin-marketcap'>${marketcap.toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
};

export default Coin;
