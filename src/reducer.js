const initialState = {
  netStatistics: [
    {
      'source': 'mnist',
      'nn_type': 'normal',
      'prediction': 1,
      'oneHotResult': [1, 2, -3, 2, 1, -2 ,3, 8, 0, 9]
    },
    {
      'source': 'web_canvas',
      'nn_type': 'normal',
      'prediction': 2,
      'oneHotResult': [9, 8, 7, 8, 9, -5, 4, 3, 2, 1]
    },
    {
      'source': 'mnist',
      'nn_type': 'conv',
      'prediction': 3,
      'oneHotResult': [2, 2, 2, 2, 2, -2, 2, 2, 2, 1]
    },
    {
      'source': 'web_canvas',
      'nn_type': 'conv',
      'prediction': 4,
      'oneHotResult': [3, 3, 3, 5, 5, -5, 5, 5, 3, 1]
    },
  ],
};

const setNetStatistics = (state, action) => {
  return Object.assign({}, state, {netStatistics: action.netStatistics});
}

function reducer(state = initialState, action) {
  switch (action.type) {
    case 'SET_NET_STATISTICS':
      return setNetStatistics(state, action);
    default:
      return state
  }
}

export default reducer;
