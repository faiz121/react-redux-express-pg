const initialState = {
  netStatistics: [1, 2, 3],
};

// const setSearchTerm = (state, action) => {
//   return Object.assign({}, state, {searchTerm: action.searchTerm});
// };

const setNetStatistics = (state, action) => {
  return Object.assign({}, state, {netStatistics: action.netStatistics});
}

// const addTodos = (state, action) => {
//   const {todos} = state;
//   return Object.assign({}, state, {todos: todos.concat(action.todo)})
// };
//
// const removeTodo = (state, action) => {
//   const {todos} = state;
//   return Object.assign({}, state, {todos: todos.filter((todo) => todo._id !== action.id)})
// };

function reducer(state = initialState, action) {
  switch (action.type) {
    case 'SET_NET_STATISTICS':
      return setNetStatistics(state, action);
    default:
      return state
  }
}

export default reducer;
