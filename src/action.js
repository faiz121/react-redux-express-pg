export function setSearchTerm(searchTerm) {
  console.log('1. action creater called');
  return { type: 'SET_SEARCH_TERM', searchTerm };
}

export function addTodo(todo) {
  return { type: 'ADD_TODO', todo };
}

export function removeTodo(id) {
  return { type: 'REMOVE_TODO', id };
}