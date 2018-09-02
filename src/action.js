import axios from 'axios';

export function setNetStatistics(netStatistics) {
  console.log('1. action creater called: ', netStatistics);
  return { type: 'SET_NET_STATISTICS', netStatistics };
}
