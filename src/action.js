import axios from 'axios';
export function setSearchTerm(searchTerm) {
  console.log('1. action creater called');
  return { type: 'SET_SEARCH_TERM', searchTerm };
}

// export function addTodo(todo) {
//   return { type: 'ADD_TODO', todo };
// }

// export function getTodosFromDB() {
//   return function (dispatch, getState) {
//     console.log('in getTodosFromDB');
//     axios.get('/api/todos/')
//         .then((res) => {
//           dispatch(addTodo(res.data));
//         })
//         .catch((error) => console.error('axios error', error));
//   }
// }
