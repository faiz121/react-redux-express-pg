const initialState = {
  netStatistics: [1, 2, 3],
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
