import axios from 'axios';
export function setNetStatistics(netStatistics) {
  console.log('1. action creater called: ', netStatistics);
  return { type: 'SET_NET_STATISTICS', netStatistics };
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
