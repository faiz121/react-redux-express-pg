const initialState = {
  searchTerm: '',
  todos: [],
  image: []
};

const setSearchTerm = (state, action) => {
  return Object.assign({}, state, {searchTerm: action.searchTerm});
};

const addTodos = (state, action) => {
  const {todos} = state;
  return Object.assign({}, state, {todos: todos.concat(action.todo)})
};

const removeTodo = (state, action) => {
  const {todos} = state;
  return Object.assign({}, state, {todos: todos.filter((todo) => todo._id !== action.id)})
};

function reducer(state = initialState, action) {
  switch (action.type) {
    case 'SET_SEARCH_TERM':
      return setSearchTerm(state, action);
    case 'ADD_TODO':
      return addTodos(state, action);
    case 'REMOVE_TODO':
      return removeTodo(state, action);
    default:
      return state
  }
}

export default reducer;
